// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.

// Developer: Jiarong Lin  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "ceres_icp.hpp"
#include "common.h"

#define DISTORTION 0

int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 0;
bool systemInited = false;

double time_edge_point = 0;
double time_plane_point = 0;
double time_full_point = 0;

size_t                                          CERES_ITERATION = 40;
size_t                                          ICP_ITERATION = 1;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_edge_point(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_plane_point(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr point_cloud_edge_point(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr point_cloud_plane_point(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr point_cloud_edge_point_last(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr point_cloud_plane_point_last(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr point_cloud_full_point(new pcl::PointCloud<PointType>());

int point_cloud_edge_point_last_num = 0;
int point_cloud_plane_point_last_num = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_incre(para_q);
Eigen::Map<Eigen::Vector3d> t_incre(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex                                   mBuf;


// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_incre);
    Eigen::Vector3d t_point_last = s * t_incre;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);


    ros::Subscriber sub_edge_point = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber sub_planar_point = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber sub_full_point = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() &&
            !surfFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            time_edge_point = cornerSharpBuf.front()->header.stamp.toSec();
            time_plane_point = surfFlatBuf.front()->header.stamp.toSec();
            time_full_point = fullPointsBuf.front()->header.stamp.toSec();

            if (time_edge_point != time_full_point ||
                time_plane_point != time_full_point)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();

            point_cloud_edge_point->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *point_cloud_edge_point);
            cornerSharpBuf.pop();

            point_cloud_plane_point->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *point_cloud_plane_point);
            surfFlatBuf.pop();

            point_cloud_full_point->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *point_cloud_full_point);
            fullPointsBuf.pop();
            
            mBuf.unlock();

            // If enable predict ?
            // q_incre.setIdentity();
            // t_incre = t_incre*0.0;
            
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                int cornerPointsSharpNum = point_cloud_edge_point->points.size();
                int surfPointsFlatNum = point_cloud_plane_point->points.size();

                for (size_t opti_counter = 0; opti_counter < ICP_ITERATION; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    // find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(point_cloud_edge_point->points[i]), &pointSel);
                        kdtree_edge_point->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(point_cloud_edge_point_last->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)point_cloud_edge_point_last->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(point_cloud_edge_point_last->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(point_cloud_edge_point_last->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (point_cloud_edge_point_last->points[j].x - pointSel.x) *
                                                        (point_cloud_edge_point_last->points[j].x - pointSel.x) +
                                                    (point_cloud_edge_point_last->points[j].y - pointSel.y) *
                                                        (point_cloud_edge_point_last->points[j].y - pointSel.y) +
                                                    (point_cloud_edge_point_last->points[j].z - pointSel.z) *
                                                        (point_cloud_edge_point_last->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(point_cloud_edge_point_last->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(point_cloud_edge_point_last->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (point_cloud_edge_point_last->points[j].x - pointSel.x) *
                                                        (point_cloud_edge_point_last->points[j].x - pointSel.x) +
                                                    (point_cloud_edge_point_last->points[j].y - pointSel.y) *
                                                        (point_cloud_edge_point_last->points[j].y - pointSel.y) +
                                                    (point_cloud_edge_point_last->points[j].z - pointSel.z) *
                                                        (point_cloud_edge_point_last->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(point_cloud_edge_point->points[i].x,
                                                       point_cloud_edge_point->points[i].y,
                                                       point_cloud_edge_point->points[i].z);
                            Eigen::Vector3d last_point_a(point_cloud_edge_point_last->points[closestPointInd].x,
                                                         point_cloud_edge_point_last->points[closestPointInd].y,
                                                         point_cloud_edge_point_last->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(point_cloud_edge_point_last->points[minPointInd2].x,
                                                         point_cloud_edge_point_last->points[minPointInd2].y,
                                                         point_cloud_edge_point_last->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                s = (point_cloud_edge_point->points[i].intensity - int(point_cloud_edge_point->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = ceres_icp_point2line<double>::Create(curr_point, last_point_a, last_point_b, 
                                                                                                Eigen::Matrix<double, 4, 1>( 1.0, 0.0, 0.0, 0.0 ),
                                                                                                Eigen::Matrix<double, 3, 1>( 0.0, 0.0, 0.0 ) );
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(point_cloud_plane_point->points[i]), &pointSel);
                        kdtree_plane_point->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(point_cloud_plane_point_last->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)point_cloud_plane_point_last->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(point_cloud_plane_point_last->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (point_cloud_plane_point_last->points[j].x - pointSel.x) *
                                                        (point_cloud_plane_point_last->points[j].x - pointSel.x) +
                                                    (point_cloud_plane_point_last->points[j].y - pointSel.y) *
                                                        (point_cloud_plane_point_last->points[j].y - pointSel.y) +
                                                    (point_cloud_plane_point_last->points[j].z - pointSel.z) *
                                                        (point_cloud_plane_point_last->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(point_cloud_plane_point_last->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(point_cloud_plane_point_last->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(point_cloud_plane_point_last->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (point_cloud_plane_point_last->points[j].x - pointSel.x) *
                                                        (point_cloud_plane_point_last->points[j].x - pointSel.x) +
                                                    (point_cloud_plane_point_last->points[j].y - pointSel.y) *
                                                        (point_cloud_plane_point_last->points[j].y - pointSel.y) +
                                                    (point_cloud_plane_point_last->points[j].z - pointSel.z) *
                                                        (point_cloud_plane_point_last->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(point_cloud_plane_point_last->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(point_cloud_plane_point_last->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(point_cloud_plane_point->points[i].x,
                                                            point_cloud_plane_point->points[i].y,
                                                            point_cloud_plane_point->points[i].z);
                                Eigen::Vector3d last_point_a(point_cloud_plane_point_last->points[closestPointInd].x,
                                                                point_cloud_plane_point_last->points[closestPointInd].y,
                                                                point_cloud_plane_point_last->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(point_cloud_plane_point_last->points[minPointInd2].x,
                                                                point_cloud_plane_point_last->points[minPointInd2].y,
                                                                point_cloud_plane_point_last->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(point_cloud_plane_point_last->points[minPointInd3].x,
                                                                point_cloud_plane_point_last->points[minPointInd3].y,
                                                                point_cloud_plane_point_last->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (point_cloud_plane_point->points[i].intensity - int(point_cloud_plane_point->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = ceres_icp_point2plane<double>::Create(curr_point, last_point_a, last_point_b, last_point_c, 
                                                                                                Eigen::Matrix<double, 4, 1>( 1.0, 0.0, 0.0, 0.0 ),
                                                                                                Eigen::Matrix<double, 3, 1>( 0.0, 0.0, 0.0 ));
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = CERES_ITERATION;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                 }
                std::cout << "--------------------------------------" << std::endl;
                std::cout << "Q_curr:" << q_w_curr.coeffs().transpose() << std::endl;
                std::cout << "T_curr:" << t_w_curr.transpose() << std::endl;
                // m_t_w_curr = m_q_w_last * t_w_incre + m_t_w_last;
                // m_q_w_curr = m_q_w_last * q_w_incre;
                t_w_curr = t_w_curr + q_w_curr * t_incre;
                q_w_curr = q_w_curr * q_incre;
            }


            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(time_plane_point);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            *point_cloud_plane_point_last = *point_cloud_plane_point;
            *point_cloud_edge_point_last = *point_cloud_edge_point;

            point_cloud_edge_point_last_num = point_cloud_edge_point_last->points.size();
            point_cloud_plane_point_last_num = point_cloud_plane_point_last->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtree_edge_point->setInputCloud(point_cloud_edge_point_last);
            kdtree_plane_point->setInputCloud(point_cloud_plane_point_last);

            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*point_cloud_edge_point_last, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(time_plane_point);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*point_cloud_plane_point_last, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(time_plane_point);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*point_cloud_full_point, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(time_plane_point);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}
