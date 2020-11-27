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

#ifndef LASER_MAPPING_HPP
#define LASER_MAPPING_HPP

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <future>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <thread>
#include <vector>

#include "cell_map_keyframe.hpp"
#include "ceres_pose_graph_3d.hpp"
#include "common.h"
#include "custom_point_cloud_interface.hpp"
#include "lidar_agent.hpp"
#include "point_cloud_registration.hpp"
#include "scene_alignment.hpp"
#include "mul_lidar_management.hpp"
#include "tools/pcl_tools.hpp"
#include "tools/tools_logger.hpp"
#include "tools/tools_ros.hpp"
#include "tools/tools_timer.hpp"
#include "ekf_pose_6d.hpp"
#define PUB_SURROUND_PTS 1
#define PCD_SAVE_RAW 1
#define PUB_DEBUG_INFO 1
#define IF_PUBLISH_SURFACE_AND_CORNER_PTS 1
#define IF_PUBLISH_MATHCING_BUFFER 0


using namespace PCL_TOOLS;
using namespace Common_tools;

class Point_cloud_registration;

class Laser_mapping
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int m_current_frame_index = 0;
    int m_para_min_match_blur = 0.0;
    int m_para_max_match_blur = 0.3;
    int m_max_buffer_size = 50000000;

    int m_mapping_init_accumulate_frames = 100;
    int m_kmean_filter_threshold = 2.0;

    double m_time_pc_corner_past = 0;
    double m_time_pc_surface_past = 0;
    double m_time_pc_full = 0;
    double m_time_odom = 0;
    double m_last_time_stamp = 0;
    double m_minimum_pt_time_stamp = 0;
    double m_maximum_pt_time_stamp = 1.0;
    float  m_last_max_blur = 0.0;

    int    m_odom_mode;
    int    m_matching_mode = 0;
    int    m_if_input_downsample_mode = 1;
    int    m_if_use_kalman_filter_prediction = 0;
    int    m_if_multiple_lidar = 0;
    int    m_maximum_parallel_thread;
    int    m_maximum_mapping_buff_thread = 1; // Maximum number of thead for matching buffer update
    int    m_maximum_history_size = 100;
    int    m_para_threshold_cell_revisit = 0;
    float  m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float  m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float  m_max_final_cost = 100.0;
    int    m_para_icp_max_iterations = 20;
    int    m_para_cere_max_iterations = 100;
    int    m_para_optimization_maximum_residual_block = 1e5;
    double m_minimum_icp_R_diff = 0.01;
    double m_minimum_icp_T_diff = 0.01;

    string m_pcd_save_dir_name, m_log_save_dir_name, m_loop_save_dir_name;

    std::list<pcl::PointCloud<PointType>> m_laser_cloud_corner_for_keyframe;
    std::list<pcl::PointCloud<PointType>> m_laser_cloud_surface_for_keyframe;

    // std::list<pcl::PointCloud<PointType>> m_laser_cloud_corner_history;
    // std::list<pcl::PointCloud<PointType>> m_laser_cloud_surface_history;

    std::list<pcl::PointCloud<PointType>> m_laser_cloud_full_history;
    std::list<double>                     m_his_reg_error;
    Eigen::Quaterniond                    m_last_his_add_q;
    Eigen::Vector3d                       m_last_his_add_t;

    //
    std::map<int, float> m_map_life_time_corner;
    std::map<int, float> m_map_life_time_surface;

    // ouput: all visualble cube points
    pcl::PointCloud<PointType>::Ptr m_laser_cloud_surround;

    // surround points in map to build tree
    int m_if_mapping_updated_corner = true;
    int m_if_mapping_updated_surface = true;


    //input & output: points in one frame. local --> global
    // input: from odom

    //kd-tree

    int m_laser_cloud_valid_Idx[ 1024 ];
    int m_laser_cloud_surround_Idx[ 1024 ];

    std::map<double, std::shared_ptr<Data_pair>> m_map_data_pair;
    std::queue<std::shared_ptr<Data_pair>>       m_queue_avail_data;

    std::queue<nav_msgs::Odometry::ConstPtr> m_odom_que;
    std::mutex                               m_mutex_buf;

    float                                     m_line_resolution = 0;
    float                                     m_plane_resolution = 0;
    pcl::VoxelGrid<PointType>                 m_down_sample_filter_corner;
    pcl::VoxelGrid<PointType>                 m_down_sample_filter_surface;

    std::vector<int>   m_point_search_Idx;
    std::vector<float> m_point_search_sq_dis;

    nav_msgs::Path m_laser_path_after_mapped_vec[MAXIMUM_LIDAR_SIZE], m_laser_path_filtered_vec[MAXIMUM_LIDAR_SIZE], m_laser_path_after_loopclosure_vec[MAXIMUM_LIDAR_SIZE];

    int                    m_if_save_to_pcd_files = 1;
    PCL_point_cloud_to_pcd m_pcl_tools_aftmap;
    PCL_point_cloud_to_pcd m_pcl_tools_raw;

    Common_tools::File_logger m_logger_common;
    Common_tools::File_logger m_logger_loop_closure;
    Common_tools::File_logger m_logger_pcd;
    Common_tools::File_logger m_logger_timer;
    Common_tools::File_logger m_logger_matching_buff;
    Scene_alignment<float>    m_sceene_align;
    Common_tools::Timer       m_timer;

    Scene_alignment<float>   m_scene_align;
    std::vector<Lidar_agent> m_lidar_agent;

    ros::Publisher              m_pub_laser_cloud_surround, m_pub_laser_cloud_map, m_pub_laser_cloud_full_res, m_pub_odom_aft_mapped, m_pub_odom_aft_filtered, m_pub_odom_aft_mapped_hight_frec;
    std::vector<ros::Publisher> m_pub_path_laser_aft_mapped_vec, m_pub_path_laser_filtered_vec, m_pub_path_laser_aft_loopclosure_vec;
    std::vector<ros::Publisher> m_maker_vec;
    ros::NodeHandle             m_ros_node_handle;
    ros::Subscriber             m_sub_laser_cloud_corner_last, m_sub_laser_cloud_surf_last, m_sub_laser_cloud_full_res, m_sub_laser_odom;

    ceres::Solver::Summary m_final_opt_summary;
    //std::list<std::thread* > m_thread_pool;
    std::list<std::future<int> *>  m_thread_pool;
    std::list<std::future<void> *> m_thread_match_buff_refresh;

    double m_maximum_in_fov_angle;
    double m_maximum_pointcloud_delay_time;
    double m_maximum_search_range_corner;
    double m_maximum_search_range_surface;
    double m_surround_pointcloud_resolution;
    double m_lastest_pc_reg_time = -3e8;
    double m_lastest_pc_matching_refresh_time = -3e8;
    double m_lastest_pc_income_time = -3e8;
    double m_degenerate_threshold = 1e-6;
    int    m_maximum_degenerate_direction = 2;
    int     m_para_if_force_update_buffer_for_matching = 0;

    std::mutex m_mutex_querypointcloud;
    std::mutex m_mutex_thread_pool;
    std::mutex m_mutex_ros_pub;
    std::mutex m_mutex_dump_full_history;
    std::mutex m_mutex_keyframe;

    float                   m_pt_cell_resolution = 1.0;
    Points_cloud_map<float> m_pt_cell_map_full;
    int                m_down_sample_replace = 1;
    std::vector<ros::Publisher> m_pub_full_point_cloud_vector;
    ros::Publisher     m_pub_last_corner_pts, m_pub_last_surface_pts;
    ros::Publisher     m_pub_match_corner_pts, m_pub_match_surface_pts, m_pub_debug_pts, m_pub_pc_aft_loop;
    std::future<void> *m_mapping_refresh_service_corner, *m_mapping_refresh_service_surface, *m_mapping_refresh_service; // Thread for mapping update
    std::future<void> *m_service_pub_surround_pts, *m_service_loop_detection;                                            // Thread for loop detection and publish surrounding pts

    Common_tools::Timer timer_all;
    std::mutex          timer_log_mutex;

    int    m_if_maps_incre_update_mean_and_cov;
    int    m_loop_closure_if_enable;
    int    m_loop_closure_if_dump_keyframe_data;
    int    m_loop_closure_minimum_keyframe_differen;
    int    m_para_scans_of_each_keyframe = 0;
    int    m_para_scans_between_two_keyframe = 0;
    int    m_para_scene_alignments_maximum_residual_block;
    int    m_if_load_extrinsic = 0;
    double m_mapping_feature_downsample_scale = 3;
    int    m_para_pub_path_downsample_factor = 10;

    int m_loop_closure_map_alignment_maximum_icp_iteration;
    int m_loop_closure_map_alignment_if_dump_matching_result;
    int m_loop_closure_maximum_keyframe_in_wating_list;
    int m_minimum_overlap_cells_num;

    float m_loop_closure_minimum_similarity_linear;
    float m_loop_closure_minimum_similarity_planar;
    float m_loop_closure_map_alignment_resolution;
    float m_loop_closure_map_alignment_inlier_threshold;

    // ANCHOR class mapping define
    //std::shared_ptr<Maps_keyframe<float>>            m_current_keyframe;
    std::list<std::shared_ptr<Maps_keyframe<float>>> m_keyframe_of_updating_list;
    std::list<std::shared_ptr<Maps_keyframe<float>>> m_keyframe_need_processing_list;
    Mul_lidar_management                             m_mul_lidar_management;
    Global_ekf                                       m_global_ekf;
    std::vector<pcl::PointCloud<pcl::PointXYZI>>     m_dump_full_pc_vector;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    int  if_pt_in_fov( const Eigen::Matrix<double, 3, 1> &pt, int lidar_id = 0 );

    void update_buff_for_matching(int lidar_id);
    void service_update_buff_for_matching();
    Laser_mapping();
    ~Laser_mapping();

    std::shared_ptr<Data_pair> get_data_pair( const double &time_stamp );

    template <typename T>
    T get_ros_parameter( ros::NodeHandle &nh, const std::string parameter_name, T &parameter, T default_val )
    {
        nh.param<T>( parameter_name.c_str(), parameter, default_val );
        ENABLE_SCREEN_PRINTF;
        screen_out << "[Laser_mapping_ros_param]: " << parameter_name << " ==> " << parameter << std::endl;
        return parameter;
    }

    void init_parameters( ros::NodeHandle &nh );

    void laserCloudCornerLastHandler( Loam_livox_custom_point_cloud laserCloudCornerLast2 );
    void laserCloudSurfLastHandler( Loam_livox_custom_point_cloud laserCloudSurfLast2 );

    void laserCloudFullResHandler( Loam_livox_custom_point_cloud laserCloudFullRes2 );

    void laserOdometryHandler( const nav_msgs::Odometry::ConstPtr &laserOdometry );

    void dump_pose_and_regerror( std::string file_name, Eigen::Quaterniond &q_curr,
                                 Eigen::Vector3d &  t_curr,
                                 std::list<double> &reg_err_vec );
    void loop_closure_pub_optimzed_path( int lidar_idx, const Ceres_pose_graph_3d::MapOfPoses &pose3d_aft_loopclosure );
    void publish_current_odometry( int lidar_id, ros::Time *timestamp );
    void loop_closure_update_buffer_for_matching( int lidar_id , const Eigen::Quaterniond &q_new, const Eigen::Vector3d &t_new );
    void service_loop_detection();
    void service_pub_surround_pts();

    Eigen::Matrix<double, 3, 1> pcl_pt_to_eigend( PointType &pt );
    void                        find_min_max_intensity( const pcl::PointCloud<PointType>::Ptr pc_ptr, float &min_I, float &max_I );
    float                       refine_blur( float in_blur, const float &min_blur, const float &max_blur );

    float compute_fov_angle( const PointType &pt );
    void  init_pointcloud_registration( Point_cloud_registration &pc_reg, int lidar_id );
    int   if_matchbuff_and_pc_sync( float point_cloud_current_timestamp );
    int   process_new_scan(std::shared_ptr<Data_pair> current_data_pair);

    void map_fusion( int src_id, int tar_id );
    void process();
    double get_average_time( Loam_livox_custom_point_cloud &full_point_cloud_msg );

    template <typename T, typename TT>
    static void save_mat_to_json_writter( T &writer, const std::string &name, const TT &eigen_mat )
    {
        writer.Key( name.c_str() ); // output a key,
        writer.StartArray();        // Between StartArray()/EndArray(),
        for ( size_t i = 0; i < ( size_t )( eigen_mat.cols() * eigen_mat.rows() ); i++ )
            writer.Double( eigen_mat( i ) );
        writer.EndArray();
    }

    template <typename T, typename TT>
    static void save_quaternion_to_json_writter( T &writer, const std::string &name, const Eigen::Quaternion<TT> &q_curr )
    {
        writer.Key( name.c_str() );
        writer.StartArray();
        writer.Double( q_curr.w() );
        writer.Double( q_curr.x() );
        writer.Double( q_curr.y() );
        writer.Double( q_curr.z() );
        writer.EndArray();
    }

    template <typename T, typename TT>
    static void save_data_vec_to_json_writter( T &writer, const std::string &name, TT &data_vec )
    {
        writer.Key( name.c_str() );
        writer.StartArray();
        for ( auto it = data_vec.begin(); it != data_vec.end(); it++ )
        {
            writer.Double( *it );
        }
        writer.EndArray();
    }
    
    void set_EFK_extrinsic(int id = -1);
};

#endif // LASER_MAPPING_HPP
