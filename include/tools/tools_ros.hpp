
#pragma once
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
namespace Common_tools
{
template <typename T, typename TT, typename TTT>
void eigen_RT_to_ros_pose( T                        eigen_q, // Eigen base quadternion
                           TT &eigen_t, // Eigen T
                           TTT &                          ros_pose )
{

    ros_pose.orientation.x = eigen_q.x();
    ros_pose.orientation.y = eigen_q.y();
    ros_pose.orientation.z = eigen_q.z();
    ros_pose.orientation.w = eigen_q.w();

    ros_pose.position.x = eigen_t( 0 );
    ros_pose.position.y = eigen_t( 1 );
    ros_pose.position.z = eigen_t( 2 );
};
}; // namespace Common_tools