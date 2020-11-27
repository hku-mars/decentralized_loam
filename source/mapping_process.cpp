#include "laser_mapping.hpp"

int    g_if_undistore = 0;
int    g_if_checkout_odometry = 0;
double history_add_t_step = 0.00;
double history_add_angle_step = 0.00;

int if_motion_deblur = 0;

void Data_pair::add_pc_corner( Loam_livox_custom_point_cloud &ros_pc )
{
    m_pc_corner = ros_pc;
    m_has_pc_corner = true;
}

void Data_pair::add_pc_plane( Loam_livox_custom_point_cloud &ros_pc )
{
    m_pc_plane = ros_pc;
    m_has_pc_plane = true;
}

void Data_pair::add_pc_full( Loam_livox_custom_point_cloud &ros_pc )
{
    m_pc_full = ros_pc;
    m_has_pc_full = true;
}

void Data_pair::add_odom( const nav_msgs::Odometry::ConstPtr &laserOdometry )
{
    m_odometry_R.x() = laserOdometry->pose.pose.orientation.x;
    m_odometry_R.y() = laserOdometry->pose.pose.orientation.y;
    m_odometry_R.z() = laserOdometry->pose.pose.orientation.z;
    m_odometry_R.w() = laserOdometry->pose.pose.orientation.w;
    m_odometry_T.x() = laserOdometry->pose.pose.position.x;
    m_odometry_T.y() = laserOdometry->pose.pose.position.y;
    m_odometry_T.z() = laserOdometry->pose.pose.position.z;
    m_has_odom_message = true;
}

bool Data_pair::is_completed()
{
    if ( g_if_checkout_odometry )
        return ( m_has_pc_corner & m_has_pc_full & m_has_pc_plane & m_has_odom_message );
    else
        return ( m_has_pc_corner & m_has_pc_full & m_has_pc_plane );
};

int Laser_mapping::if_pt_in_fov( const Eigen::Matrix<double, 3, 1> &pt, int  lidar_id  )
{
 
    auto pt_affine = m_lidar_agent[ lidar_id ].m_q_w_curr.inverse() * ( pt - m_lidar_agent[ lidar_id ].m_t_w_curr );

    if ( m_maximum_in_fov_angle < 180.0 )
    {
        if ( pt_affine( 0 ) < 0 )
            return 0;
    }
    float angle = Eigen_math::vector_angle( pt_affine, Eigen::Matrix<double, 3, 1>( 1, 0, 0 ), 1 );

    if ( angle * 57.3 < m_maximum_in_fov_angle )
        return 1;
    else
        return 0;
}

void Laser_mapping::init_pointcloud_registration( Point_cloud_registration &pc_reg, int lidar_id )
{
    //printf_line;
    //pc_reg.m_kdtree_corner_from_map = m_kdtree_corner_from_map;
    //pc_reg.m_kdtree_surf_from_map = m_kdtree_surf_from_map;
    pc_reg.m_logger_common = &m_logger_common;
    pc_reg.m_logger_pcd = &m_logger_pcd;
    pc_reg.m_logger_timer = &m_logger_timer;
    pc_reg.m_timer = &m_timer;
    pc_reg.m_if_motion_deblur = if_motion_deblur;
    pc_reg.m_current_frame_index = m_current_frame_index;
    pc_reg.m_mapping_init_accumulate_frames = m_mapping_init_accumulate_frames;
    
    
    pc_reg.m_maximum_degenerate_direction = m_maximum_degenerate_direction;

    pc_reg.m_last_time_stamp = m_last_time_stamp;
    pc_reg.m_para_max_angular_rate = m_para_max_angular_rate;
    pc_reg.m_para_max_speed = m_para_max_speed;
    pc_reg.m_max_final_cost = m_max_final_cost;
    pc_reg.m_para_icp_max_iterations = m_para_icp_max_iterations;
    pc_reg.m_para_cere_max_iterations = m_para_cere_max_iterations;
    pc_reg.m_maximum_allow_residual_block = m_para_optimization_maximum_residual_block;
    pc_reg.m_minimum_pt_time_stamp = m_minimum_pt_time_stamp;
    pc_reg.m_maximum_pt_time_stamp = m_maximum_pt_time_stamp;
    pc_reg.m_minimum_icp_R_diff = m_minimum_icp_R_diff;
    pc_reg.m_minimum_icp_T_diff = m_minimum_icp_T_diff;
    pc_reg.m_q_w_last = m_lidar_agent[ lidar_id ].m_q_w_curr;
    pc_reg.m_t_w_last = m_lidar_agent[ lidar_id ].m_t_w_curr;

    pc_reg.m_q_w_curr = m_lidar_agent[ lidar_id ].m_q_w_curr;
    pc_reg.m_t_w_curr = m_lidar_agent[ lidar_id ].m_t_w_curr;

    if ( lidar_id == 0 )
    {
        pc_reg.m_maximum_allow_residual_block *= 3.0;
    }
    if ( m_mul_lidar_management.m_if_have_merge[ lidar_id ] == 0 )
    {
        pc_reg.m_degenerate_threshold = m_degenerate_threshold / 100.0;
        // pc_reg.m_maximum_degenerate_direction = 0;
    }
    else
    {
        pc_reg.m_degenerate_threshold = m_degenerate_threshold;
    }
    //printf_line;
};

// ANCHOR ros_parameter_setting
void Laser_mapping::init_parameters( ros::NodeHandle &nh )
{

    get_ros_parameter<float>( nh, "feature_extraction/mapping_line_resolution", m_line_resolution, 0.4 );
    get_ros_parameter<float>( nh, "feature_extraction/mapping_plane_resolution", m_plane_resolution, 0.8 );

    if ( m_odom_mode == 1 )
    {
        //m_max_buffer_size = 3e8;
    }

    get_ros_parameter<int>( nh, "common/if_verbose_screen_printf", m_if_verbose_screen_printf, 1 );
    get_ros_parameter<int>( nh, "common/odom_mode", m_odom_mode, 0 );
    get_ros_parameter<int>( nh, "common/maximum_parallel_thread", m_maximum_parallel_thread, 2 );
    get_ros_parameter<int>( nh, "common/if_motion_deblur", if_motion_deblur, 0 );
    get_ros_parameter<int>( nh, "common/if_save_to_pcd_files", m_if_save_to_pcd_files, 0 );
    get_ros_parameter<int>( nh, "common/if_update_mean_and_cov_incrementally", m_if_maps_incre_update_mean_and_cov, 0 );
    get_ros_parameter<int>( nh, "common/threshold_cell_revisit", m_para_threshold_cell_revisit, 5000 );
    get_ros_parameter<int>( nh, "common/if_multiple_lidar", m_if_multiple_lidar, 0 );
    
    get_ros_parameter<double>( nh, "optimization/minimum_icp_R_diff", m_minimum_icp_R_diff, 0.01 );
    get_ros_parameter<double>( nh, "optimization/minimum_icp_T_diff", m_minimum_icp_T_diff, 0.01 );
    get_ros_parameter<int>( nh, "optimization/icp_maximum_iteration", m_para_icp_max_iterations, 20 );
    get_ros_parameter<int>( nh, "optimization/ceres_maximum_iteration", m_para_cere_max_iterations, 20 );
    get_ros_parameter<int>( nh, "optimization/maximum_residual_blocks", m_para_optimization_maximum_residual_block, 1e5 );
    get_ros_parameter<float>( nh, "optimization/max_allow_incre_R", m_para_max_angular_rate, 200.0 / 50.0 );
    get_ros_parameter<float>( nh, "optimization/max_allow_incre_T", m_para_max_speed, 100.0 / 50.0 );
    get_ros_parameter<float>( nh, "optimization/max_allow_final_cost", m_max_final_cost, 1.0 );
    get_ros_parameter<int>( nh, "optimization/if_use_kalman_filter_prediction", m_if_use_kalman_filter_prediction, 0 );
    get_ros_parameter<double>( nh, "optimization/threshold_degenerate", m_degenerate_threshold, 1e-6 );
    get_ros_parameter<int>( nh, "optimization/degenerate_direction", m_maximum_degenerate_direction, 2 );

    get_ros_parameter<int>( nh, "mapping/init_accumulate_frames", m_mapping_init_accumulate_frames, 50 );
    get_ros_parameter<int>( nh, "mapping/maximum_histroy_buffer", m_maximum_history_size, 100 );
    get_ros_parameter<int>( nh, "mapping/maximum_mapping_buffer", m_max_buffer_size, 5 );
    get_ros_parameter<int>( nh, "mapping/matching_mode", m_matching_mode, 1 );
    get_ros_parameter<int>( nh, "mapping/input_downsample_mode", m_if_input_downsample_mode, 1 );
    get_ros_parameter<int>( nh, "mapping/matching_force_update", m_para_if_force_update_buffer_for_matching, 0 );
    get_ros_parameter<double>( nh, "mapping/maximum_in_fov_angle", m_maximum_in_fov_angle, 30 );
    get_ros_parameter<double>( nh, "mapping/maximum_pointcloud_delay_time", m_maximum_pointcloud_delay_time, 0.1 );
    get_ros_parameter<double>( nh, "mapping/maximum_in_fov_angle", m_maximum_in_fov_angle, 30 );
    get_ros_parameter<double>( nh, "mapping/maximum_search_range_corner", m_maximum_search_range_corner, 100 );
    get_ros_parameter<double>( nh, "mapping/maximum_search_range_surface", m_maximum_search_range_surface, 100 );
    get_ros_parameter<double>( nh, "mapping/surround_pointcloud_resolution", m_surround_pointcloud_resolution, 0.5 );
    get_ros_parameter<double>( nh, "mapping/feature_downsample_scale", m_mapping_feature_downsample_scale, 1.0 );
    get_ros_parameter<int>( nh, "mapping/pub_path_downsample_scale", m_para_pub_path_downsample_factor, 1 );
    get_ros_parameter<int>( nh, "mapping/if_load_extrinsic", m_if_load_extrinsic, 1 );


    get_ros_parameter<int>( nh, "lidar_fusion/minimum_overlap_cell_number", m_minimum_overlap_cells_num, 40 );

    get_ros_parameter<int>( nh, "loop_closure/if_enable_loop_closure", m_loop_closure_if_enable, 0 );
    get_ros_parameter<int>( nh, "loop_closure/minimum_keyframe_differen", m_loop_closure_minimum_keyframe_differen, 200 );
    get_ros_parameter<float>( nh, "loop_closure/minimum_similarity_linear", m_loop_closure_minimum_similarity_linear, 0.65 );
    get_ros_parameter<float>( nh, "loop_closure/minimum_similarity_planar", m_loop_closure_minimum_similarity_planar, 0.95 );
    get_ros_parameter<float>( nh, "loop_closure/map_alignment_resolution", m_loop_closure_map_alignment_resolution, 0.2 );
    get_ros_parameter<float>( nh, "loop_closure/map_alignment_inlier_threshold", m_loop_closure_map_alignment_inlier_threshold, 0.35 );
    get_ros_parameter<int>( nh, "loop_closure/map_alignment_maximum_icp_iteration", m_loop_closure_map_alignment_maximum_icp_iteration, 2 );
    get_ros_parameter<int>( nh, "loop_closure/maximum_keyframe_in_waiting_list", m_loop_closure_maximum_keyframe_in_wating_list, 3 );
    get_ros_parameter<int>( nh, "loop_closure/scans_of_each_keyframe", m_para_scans_of_each_keyframe, 300 );
    get_ros_parameter<int>( nh, "loop_closure/scans_between_two_keyframe", m_para_scans_between_two_keyframe, 100 );
    get_ros_parameter<int>( nh, "loop_closure/scene_alignment_maximum_residual_block", m_para_scene_alignments_maximum_residual_block, 5000 );
    get_ros_parameter<int>( nh, "loop_closure/if_dump_keyframe_data", m_loop_closure_if_dump_keyframe_data, 0 );
    get_ros_parameter<int>( nh, "loop_closure/map_alignment_if_dump_matching_result", m_loop_closure_map_alignment_if_dump_matching_result, 0 );

    get_ros_parameter<std::string>( nh, "common/log_save_dir", m_log_save_dir_name, "../" );
    m_pt_cell_map_full.set_update_mean_and_cov_incrementally( m_if_maps_incre_update_mean_and_cov );

    m_logger_common.set_log_dir( m_log_save_dir_name );
    m_logger_common.init( "mapping.log" );
    m_logger_timer.set_log_dir( m_log_save_dir_name );
    m_logger_timer.init( "timer.log" );
    m_logger_matching_buff.set_log_dir( m_log_save_dir_name );
    m_logger_matching_buff.init( "match_buff.log" );

    get_ros_parameter<std::string>( nh, "common/pcd_save_dir", m_pcd_save_dir_name, std::string( "./" ) );
    get_ros_parameter<std::string>( nh, "common/loop_save_dir", m_loop_save_dir_name, m_pcd_save_dir_name.append( "_loop" ) );
    m_sceene_align.init( m_loop_save_dir_name );

    if ( 1 )
    {
        m_pcl_tools_aftmap.set_save_dir_name( m_pcd_save_dir_name );
        m_pcl_tools_raw.set_save_dir_name( m_pcd_save_dir_name );
    }

    m_logger_pcd.set_log_dir( m_log_save_dir_name );
    m_logger_pcd.init( "poses.log" );

    LOG_FILE_LINE( m_logger_common );
    *m_logger_common.get_ostream() << m_logger_common.version();

    screen_printf( "line resolution %f plane resolution %f \n", m_line_resolution, m_plane_resolution );
    m_logger_common.printf( "line resolution %f plane resolution %f \n", m_line_resolution, m_plane_resolution );
    m_down_sample_filter_corner.setLeafSize( m_line_resolution, m_line_resolution, m_line_resolution );
    m_down_sample_filter_surface.setLeafSize( m_plane_resolution, m_plane_resolution, m_plane_resolution );

    m_lidar_agent.resize( MAXIMUM_LIDAR_SIZE );
    m_dump_full_pc_vector.resize(MAXIMUM_LIDAR_SIZE);
    for ( size_t i = 0; i < MAXIMUM_LIDAR_SIZE; i++ )
    {
        // printf_line;
        // Lidar_agent * new_agent = new Lidar_agent();
        // m_lidar_agent.push_back( *new_agent );
        // printf_line;
        m_lidar_agent[ i ].init_log( i );
        m_dump_full_pc_vector[i].points.reserve(1e5);
    }

    std::vector<double> lidar_extrinsic_data;
    nh.getParam( "/Lidar_extrinsic", lidar_extrinsic_data);
    m_mul_lidar_management.set_extrinsic(lidar_extrinsic_data);

    m_global_ekf.m_lidar_initial_extrinsic_R_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_R_vec[ 0 ] );
    m_global_ekf.m_lidar_initial_extrinsic_R_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_R_vec[ 3 ] );
    m_global_ekf.m_lidar_initial_extrinsic_R_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_R_vec[ 4 ] );
    m_global_ekf.m_lidar_initial_extrinsic_R_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_R_vec[ 5 ] );
    m_global_ekf.m_lidar_initial_extrinsic_R_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_R_vec[ 6 ] );

    m_global_ekf.m_lidar_initial_extrinsic_T_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_T_vec[ 0 ] );
    m_global_ekf.m_lidar_initial_extrinsic_T_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_T_vec[ 3 ] );
    m_global_ekf.m_lidar_initial_extrinsic_T_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_T_vec[ 4 ] );
    m_global_ekf.m_lidar_initial_extrinsic_T_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_T_vec[ 5 ] );
    m_global_ekf.m_lidar_initial_extrinsic_T_vec.push_back( m_mul_lidar_management.m_lidar_extrinsic_T_vec[ 6 ] );

    m_global_ekf.init();
    m_global_ekf.init_log();
    m_global_ekf.m_if_dump_all_traj = 0;
    m_global_ekf.m_play_index_vec.resize(5);
    m_global_ekf.m_lidar_agent_vec.resize(5);
    printf_line;
    m_global_ekf.m_lidar_agent_vec[0] =  &m_lidar_agent[ 0 ];
    m_global_ekf.m_lidar_agent_vec[1] =  &m_lidar_agent[ 3 ];
    m_global_ekf.m_lidar_agent_vec[2] =  &m_lidar_agent[ 4 ];
    m_global_ekf.m_lidar_agent_vec[3] =  &m_lidar_agent[ 5 ];
    m_global_ekf.m_lidar_agent_vec[4] =  &m_lidar_agent[ 6 ];
    printf_line;
    

    for(size_t i = 0 ;i < 0; i++)
    {
        m_global_ekf.m_local_lidar_ekf_io_vec.push_back( new Local_lidar_ekf( ) );
        //m_global_ekf.m_local_lidar_ekf_io_vec.back()->init_filter();
        m_global_ekf.m_play_index_vec[ i ] = 0;
    }
    printf_line;
};


void Laser_mapping::service_update_buff_for_matching()
{
    while ( 1 )
    {
        //if ( m_if_mapping_updated_corner == false and m_if_mapping_updated_surface == false )
        std::this_thread::sleep_for( std::chrono::nanoseconds( 100 ) );
        for ( size_t idx = 0; idx < MAXIMUM_LIDAR_SIZE; idx++ )
        {
            update_buff_for_matching( idx );
        }
    }
}

Laser_mapping::Laser_mapping()
{

    m_laser_cloud_surround = pcl::PointCloud<PointType>::Ptr( new pcl::PointCloud<PointType>() );

    init_parameters( m_ros_node_handle );

    //livox_corners
    m_sub_laser_cloud_corner_last = m_ros_node_handle.subscribe<Loam_livox_custom_point_cloud>( "/pc2_corners", 10000, &Laser_mapping::laserCloudCornerLastHandler, this );
    m_sub_laser_cloud_surf_last = m_ros_node_handle.subscribe<Loam_livox_custom_point_cloud>( "/pc2_surface", 10000, &Laser_mapping::laserCloudSurfLastHandler, this );
    m_sub_laser_cloud_full_res = m_ros_node_handle.subscribe<Loam_livox_custom_point_cloud>( "/pc2_full", 10000, &Laser_mapping::laserCloudFullResHandler, this );
    m_sub_laser_odom = m_ros_node_handle.subscribe<nav_msgs::Odometry>( "/laser_odom_to_init", 10000, &Laser_mapping::laserOdometryHandler, this );

    m_pub_laser_cloud_surround = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_surround", 10000 );

    m_pub_last_corner_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/features_corners", 10000 );
    m_pub_last_surface_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/features_surface", 10000 );

    m_pub_match_corner_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/match_pc_corners", 10000 );
    m_pub_match_surface_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/match_pc_surface", 10000 );
    m_pub_pc_aft_loop = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/pc_aft_loop_closure", 10000 );
    m_pub_debug_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/pc_debug", 10000 );

    m_pub_laser_cloud_map = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/laser_cloud_map", 10000 );
    m_pub_laser_cloud_full_res = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( "/full_reg_point_cloud", 10000 );
    m_pub_odom_aft_mapped = m_ros_node_handle.advertise<nav_msgs::Odometry>( "/aft_mapped_to_init", 10000 );
    m_pub_odom_aft_filtered = m_ros_node_handle.advertise<nav_msgs::Odometry>( "/odom_after_filtered", 10000 );
    m_pub_odom_aft_mapped_hight_frec = m_ros_node_handle.advertise<nav_msgs::Odometry>( "/aft_mapped_to_init_high_frec", 10000 );
    //ANCHOR  set resolution
    m_pt_cell_resolution = 1.0;
    
    for ( size_t idx = 0; idx < MAXIMUM_LIDAR_SIZE; idx++ )
    {
        m_pub_path_laser_aft_mapped_vec.push_back( m_ros_node_handle.advertise<nav_msgs::Path>( std::string( "/aft_mapped_path_" ).append( std::to_string( idx ) ), 10000 ) );
        m_pub_path_laser_aft_loopclosure_vec.push_back( m_ros_node_handle.advertise<nav_msgs::Path>( std::string( "/aft_loopclosure_path" ).append( std::to_string( idx ) ), 10000 ) );
        m_pub_path_laser_filtered_vec.push_back( m_ros_node_handle.advertise<nav_msgs::Path>( std::string( "/path_filtered" ).append( std::to_string( idx ) ), 10000 ) );
        m_pub_full_point_cloud_vector.push_back (m_ros_node_handle.advertise<sensor_msgs::PointCloud2>( std::string("/full_reg_point_cloud").append( std::to_string( idx ) ), 10000 ) );
        if ( m_matching_mode != 0 )
        {
            m_mul_lidar_management.m_pt_cell_map_corners[ idx ].set_resolution( m_pt_cell_resolution * 10.0 );
            m_mul_lidar_management.m_pt_cell_map_planes[ idx ].set_resolution( m_pt_cell_resolution * 10.0 );
        }
        else
        {
            m_mul_lidar_management.m_pt_cell_map_corners[ idx ].set_resolution( m_pt_cell_resolution );
            m_mul_lidar_management.m_pt_cell_map_planes[ idx ].set_resolution( m_pt_cell_resolution );
        }
        
        m_mul_lidar_management.m_pt_cell_map_corners[ idx ].m_minimum_revisit_threshold = m_para_threshold_cell_revisit;
        m_mul_lidar_management.m_pt_cell_map_planes[ idx ].m_minimum_revisit_threshold = m_para_threshold_cell_revisit;
    }
    if ( m_loop_closure_if_enable == 0 )
    {
        m_pt_cell_resolution = 20.0;
    }

    m_pt_cell_map_full.set_resolution( m_pt_cell_resolution );


    m_pt_cell_map_full.m_minimum_revisit_threshold = m_para_threshold_cell_revisit;

    m_keyframe_of_updating_list.push_back( std::make_shared<Maps_keyframe<float>>() );
    //m_current_keyframe = std::make_shared<Maps_keyframe<float>>();
    screen_out << "Laser_mapping init OK" << endl;
};

Laser_mapping::~Laser_mapping()
{

};

std::shared_ptr<Data_pair> Laser_mapping::get_data_pair( const double &time_stamp )
{
    std::map<double, std::shared_ptr<Data_pair>>::iterator it = m_map_data_pair.find( time_stamp );
    if ( it == m_map_data_pair.end() )
    {
        std::shared_ptr<Data_pair> date_pair_ptr = std::make_shared<Data_pair>();
        m_map_data_pair.insert( std::make_pair( time_stamp, date_pair_ptr ) );
        return date_pair_ptr;
    }
    else
    {
        return it->second;
    }
};

void Laser_mapping::laserCloudCornerLastHandler( Loam_livox_custom_point_cloud laserCloudCornerLast2 )
{
    std::unique_lock<std::mutex> lock( m_mutex_buf );
    std::shared_ptr<Data_pair>   data_pair = get_data_pair( laserCloudCornerLast2.header.stamp.toSec() );
    data_pair->add_pc_corner( laserCloudCornerLast2 );
    if ( data_pair->is_completed() )
    {
        m_queue_avail_data.push( data_pair );
    }
};

void Laser_mapping::laserCloudSurfLastHandler( Loam_livox_custom_point_cloud laserCloudSurfLast2 )
{
    std::unique_lock<std::mutex> lock( m_mutex_buf );
    std::shared_ptr<Data_pair>   data_pair = get_data_pair( laserCloudSurfLast2.header.stamp.toSec() );
    data_pair->add_pc_plane( laserCloudSurfLast2 );
    if ( data_pair->is_completed() )
    {
        m_queue_avail_data.push( data_pair );
    }
};

void Laser_mapping::laserCloudFullResHandler( Loam_livox_custom_point_cloud laserCloudFullRes2 )
{
    std::unique_lock<std::mutex> lock( m_mutex_buf );
    std::shared_ptr<Data_pair>   data_pair = get_data_pair( laserCloudFullRes2.header.stamp.toSec() );
    data_pair->add_pc_full( laserCloudFullRes2 );
    if ( data_pair->is_completed() )
    {
        m_queue_avail_data.push( data_pair );
    }
};

void Laser_mapping::laserOdometryHandler( const nav_msgs::Odometry::ConstPtr &laserOdometry )
{
    g_if_checkout_odometry = 1;
    std::unique_lock<std::mutex> lock( m_mutex_buf );
    std::shared_ptr<Data_pair>   data_pair = get_data_pair( laserOdometry->header.stamp.toSec() );
    data_pair->add_odom( laserOdometry );
    if ( data_pair->is_completed() )
    {
        m_queue_avail_data.push( data_pair );
    }
};

void Laser_mapping::dump_pose_and_regerror( std::string file_name, Eigen::Quaterniond &q_curr,
                                            Eigen::Vector3d &  t_curr,
                                            std::list<double> &reg_err_vec )
{
    rapidjson::Document                        document;
    rapidjson::StringBuffer                    sb;
    rapidjson::Writer<rapidjson::StringBuffer> writer( sb );
    writer.StartObject();
    writer.SetMaxDecimalPlaces( 1000 ); // like set_precision
    save_quaternion_to_json_writter( writer, "Q", q_curr );
    save_mat_to_json_writter( writer, "T", t_curr );
    save_data_vec_to_json_writter( writer, "Reg_err", reg_err_vec );
    writer.EndObject();
    std::fstream ofs;
    ofs.open( file_name.c_str(), std::ios_base::out );
    if ( ofs.is_open() )
    {
        ofs << std::string( sb.GetString() ).c_str();
        ofs.close();
    }
    else
    {
        for ( int i = 0; i < 100; i++ )
        {
            screen_out << "Write data to file: " << file_name << " error!!!" << std::endl;
        }
    }
};

void Laser_mapping::loop_closure_pub_optimzed_path( int lidar_id, const Ceres_pose_graph_3d::MapOfPoses &pose3d_aft_loopclosure )
{

    nav_msgs::Odometry odom;
    m_laser_path_after_loopclosure_vec[lidar_id].header.stamp = ros::Time::now();
    m_laser_path_after_loopclosure_vec[lidar_id].header.frame_id = "/camera_init";
    for ( auto it = pose3d_aft_loopclosure.begin();
          it != pose3d_aft_loopclosure.end(); it++ )
    {
        geometry_msgs::PoseStamped  pose_stamp;
        Ceres_pose_graph_3d::Pose3d pose_3d = it->second;

        Common_tools::eigen_RT_to_ros_pose( pose_3d.q, pose_3d.p, pose_stamp.pose );

        pose_stamp.header.frame_id = "/camera_init";

        m_laser_path_after_loopclosure_vec[lidar_id].poses.push_back( pose_stamp );
    }
    // 
    m_pub_path_laser_aft_loopclosure_vec[lidar_id].publish( m_laser_path_after_loopclosure_vec[lidar_id] );
};

void Laser_mapping::publish_current_odometry( int lidar_id, ros::Time *timestamp = nullptr )
{
    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = "/camera_init";
    odomAftMapped.child_frame_id = "/aft_mapped";
    if ( timestamp == nullptr )
    {
        odomAftMapped.header.stamp = ros::Time::now();
    }
    else
    {
        odomAftMapped.header.stamp = *timestamp;
    }

    Common_tools::eigen_RT_to_ros_pose( m_lidar_agent[ lidar_id ].m_q_w_curr, m_lidar_agent[ lidar_id ].m_t_w_curr, odomAftMapped.pose.pose );

    m_pub_odom_aft_mapped.publish( odomAftMapped ); // name: Odometry aft_mapped_to_init
};

void Laser_mapping::loop_closure_update_buffer_for_matching( int lidar_id, const Eigen::Quaterniond &q_new, const Eigen::Vector3d &t_new )
{
    std::cout << "Hello, this is loop_closure_update_buffer_for_matching" << std::endl;
    Eigen::Quaterniond new_tr_q = q_new * m_lidar_agent[ lidar_id ].m_q_w_curr.inverse();
    Eigen::Vector3d    new_tr_t = new_tr_q * ( -1 * m_lidar_agent[ lidar_id ].m_t_w_curr ) + t_new;
    Eigen::Matrix4d    tr_mat_44;
    tr_mat_44.setIdentity();
    tr_mat_44.block( 0, 0, 3, 3 ) = new_tr_q.toRotationMatrix();
    tr_mat_44.block( 0, 3, 3, 1 ) = new_tr_t;
    if ( m_matching_mode == 0 )
    {
        for ( auto it = m_mul_lidar_management.m_laser_cloud_corner_history_vec[ lidar_id ].begin(); it != m_mul_lidar_management.m_laser_cloud_corner_history_vec[ lidar_id ].end(); it++ )
        {
            pcl::transformPointCloud( *it, *it, tr_mat_44 );
        }

        for ( auto it = m_mul_lidar_management.m_laser_cloud_surface_history_vec[ lidar_id ].begin(); it != m_mul_lidar_management.m_laser_cloud_surface_history_vec[ lidar_id ].end(); it++ )
        {
            pcl::transformPointCloud( *it, *it, tr_mat_44 );
        }
    }
};

//ANCHOR loop_detection
void Laser_mapping::service_loop_detection()
{
    int last_update_index = 0;

    sensor_msgs::PointCloud2                           ros_laser_cloud_surround;
    pcl::PointCloud<PointType>                         pt_full;
    Eigen::Quaterniond                                 q_curr;
    Eigen::Vector3d                                    t_curr;
    std::list<double>                                  reg_error_his;
    std::string                                        json_file_name;
    int                                                curren_frame_idx;
    std::vector<std::shared_ptr<Maps_keyframe<float>>> keyframe_vec;
    Mapping_refine<PointType>                          map_rfn;
    std::vector<std::string>                           m_filename_vec;

    std::map<int, std::string>               map_file_name;
    Ceres_pose_graph_3d::MapOfPoses          pose3d_map, pose3d_map_ori;
    Ceres_pose_graph_3d::VectorOfPose        pose3d_vec;
    Ceres_pose_graph_3d::VectorOfConstraints constrain_vec;

    float avail_ratio_plane = 0.05; // 0.05 for 300 scans, 0.15 for 1000 scans
    float avail_ratio_line = 0.03;
    m_scene_align.init( m_loop_save_dir_name );
    m_scene_align.m_accepted_threshold = m_loop_closure_map_alignment_inlier_threshold;
    m_scene_align.m_maximum_icp_iteration = m_loop_closure_map_alignment_maximum_icp_iteration;
    // scene_align. =  m_
    PCL_TOOLS::PCL_point_cloud_to_pcd pcd_saver;
    pcd_saver.set_save_dir_name( std::string( m_loop_save_dir_name ).append( "/pcd" ) );
    map_rfn.set_save_dir( std::string( m_loop_save_dir_name ).append( "/mapping_refined" ) );
    map_rfn.set_down_sample_resolution( 0.2 );

    std::map<int, pcl::PointCloud<PointType>> map_id_pc;
    int                                       if_end = 0;
    pcl::VoxelGrid<PointType>                 down_sample_filter;

    m_logger_loop_closure.set_log_dir( m_log_save_dir_name );
    m_logger_loop_closure.init( "loop_closure.log" );

    down_sample_filter.setLeafSize( m_surround_pointcloud_resolution, m_surround_pointcloud_resolution, m_surround_pointcloud_resolution );
    while ( 1 )
    {
        // printf_line;
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        //m_mutex_dump_full_history.lock();

        if ( m_keyframe_need_processing_list.size() == 0 )
        {
            continue;
        }

        printf_line;

        m_timer.tic( "New keyframe" );
        q_curr = m_keyframe_need_processing_list.front()->m_pose_q;
        t_curr = m_keyframe_need_processing_list.front()->m_pose_t;
        // q_curr = m_q_w_curr;
        // t_curr = m_t_w_curr;
        reg_error_his = m_his_reg_error;

        m_keyframe_need_processing_list.front()->update_features_of_each_cells();
        m_keyframe_need_processing_list.front()->analyze();
        printf_line;
        keyframe_vec.push_back( m_keyframe_need_processing_list.front() );
        m_mutex_keyframe.lock();
        m_keyframe_need_processing_list.pop_front();
        m_mutex_keyframe.unlock();

        curren_frame_idx = keyframe_vec.back()->m_ending_frame_idx;

        down_sample_filter.setInputCloud( keyframe_vec.back()->m_accumulated_point_cloud_full.makeShared() );
        down_sample_filter.filter( keyframe_vec.back()->m_accumulated_point_cloud_full );

        map_id_pc.insert( std::make_pair( map_id_pc.size(), keyframe_vec.back()->m_accumulated_point_cloud_full ) );

        pose3d_vec.push_back( Ceres_pose_graph_3d::Pose3d( q_curr, t_curr ) );
        pose3d_map.insert( std::make_pair( pose3d_map.size(), Ceres_pose_graph_3d::Pose3d( q_curr, t_curr ) ) );

        if ( pose3d_vec.size() >= 2 )
        {
            Ceres_pose_graph_3d::Constraint3d temp_csn;
            Eigen::Vector3d                   relative_T = pose3d_vec[ pose3d_vec.size() - 2 ].q.inverse() * ( t_curr - pose3d_vec[ pose3d_vec.size() - 2 ].p );
            Eigen::Quaterniond                relative_Q = pose3d_vec[ pose3d_vec.size() - 2 ].q.inverse() * q_curr;

            temp_csn = Ceres_pose_graph_3d::Constraint3d( pose3d_vec.size() - 2, pose3d_vec.size() - 1,
                                                          relative_Q, relative_T );
            constrain_vec.push_back( temp_csn );
        }
        printf_line;

        // Save pose
        json_file_name = std::string( m_loop_save_dir_name ).append( "/pose_" ).append( std::to_string( curren_frame_idx ) ).append( ".json" );
        dump_pose_and_regerror( json_file_name, q_curr, t_curr, reg_error_his );
        last_update_index = m_current_frame_index;
        m_timer.tic( "Find loop" );

        std::shared_ptr<Maps_keyframe<float>> last_keyframe = keyframe_vec.back();

        if ( m_loop_closure_if_dump_keyframe_data ) // Dump points cloud data
        {
            json_file_name = std::string( "keyframe_" ).append( std::to_string( curren_frame_idx ) ).append( ".json" );
            last_keyframe->save_to_file( std::string( m_loop_save_dir_name ), json_file_name ); // Save keyframe data
            pcd_saver.save_to_pcd_files( "pcd", pt_full, curren_frame_idx );                    // Save to pcd files
        }
        
        map_file_name.insert( std::make_pair( map_file_name.size(), std::string( m_loop_save_dir_name ).append( "/" ).append( json_file_name ) ) );
        m_filename_vec.push_back( std::string( m_loop_save_dir_name ).append( "/" ).append( json_file_name ) );
        float sim_plane_res = 0;
        float sim_line_res = 0;
        float sim_plane_res_roi = 0, sim_line_res_roi = 0;

        m_logger_loop_closure.printf( "--- Current_idx = %d, lidar_frame_idx = %d ---\r\n", keyframe_vec.size(), curren_frame_idx );
        m_logger_loop_closure.printf( "%s", last_keyframe->get_frame_info().c_str() );

        for ( size_t his = 0; his < keyframe_vec.size() - 1; his++ )
        {
            printf_line;
            if ( if_end )
            {
                break;
            }
            cout << "keyframe_vec.size() = " << keyframe_vec.size() << ", "<< m_loop_closure_minimum_keyframe_differen << endl;
            if ( keyframe_vec.size() - his < ( size_t ) m_loop_closure_minimum_keyframe_differen )
            {
                continue;
            }
            printf_line;
            float ratio_non_zero_plane_his = keyframe_vec[ his ]->m_ratio_nonzero_plane;
            float ratio_non_zero_line_his = keyframe_vec[ his ]->m_ratio_nonzero_line;

            if ( ( ratio_non_zero_plane_his < avail_ratio_plane ) && ( ratio_non_zero_line_his < avail_ratio_line ) )
                continue;

            if ( abs( keyframe_vec[ his ]->m_roi_range - last_keyframe->m_roi_range ) > 5.0 )
            {
                continue;
            }

            sim_plane_res = last_keyframe->max_similiarity_of_two_image( last_keyframe->m_feature_img_plane, keyframe_vec[ his ]->m_feature_img_plane );
            sim_line_res = last_keyframe->max_similiarity_of_two_image( last_keyframe->m_feature_img_line, keyframe_vec[ his ]->m_feature_img_line );

            if ( ( ( sim_line_res > m_loop_closure_minimum_similarity_linear ) && ( sim_plane_res > 0.92 ) ) ||
                 ( sim_plane_res > m_loop_closure_minimum_similarity_planar ) )
            {
                if ( 0 ) // Enable check in roi
                {
                    sim_plane_res_roi = last_keyframe->max_similiarity_of_two_image( last_keyframe->m_feature_img_plane_roi, keyframe_vec[ his ]->m_feature_img_plane_roi );
                    sim_line_res_roi = last_keyframe->max_similiarity_of_two_image( last_keyframe->m_feature_img_line_roi, keyframe_vec[ his ]->m_feature_img_line_roi );
                    if ( ( ( sim_plane_res_roi > m_loop_closure_minimum_similarity_linear ) && ( sim_plane_res > 0.92 ) ) ||
                         ( sim_line_res_roi > m_loop_closure_minimum_similarity_planar ) )
                    {
                        m_logger_loop_closure.printf( "Range in roi check pass\r\n" );
                    }
                    else
                    {
                        continue;
                    }
                }

                if ( ( last_keyframe->m_set_cell.size() - keyframe_vec[ his ]->m_set_cell.size() ) / ( last_keyframe->m_set_cell.size() + keyframe_vec[ his ]->m_set_cell.size() ) * 0.1 )
                {
                    continue;
                }

                m_scene_align.set_downsample_resolution( m_loop_closure_map_alignment_resolution, m_loop_closure_map_alignment_resolution );
                m_scene_align.m_para_scene_alignments_maximum_residual_block = m_para_scene_alignments_maximum_residual_block;
                double icp_score = m_scene_align.find_tranfrom_of_two_mappings( last_keyframe, keyframe_vec[ his ], m_loop_closure_map_alignment_if_dump_matching_result );

                screen_printf( "===============================================\r\n" );
                screen_printf( "%s -- %s\r\n", m_filename_vec[ keyframe_vec.size() - 1 ].c_str(), m_filename_vec[ his ].c_str() );
                screen_printf( "ICP inlier threshold = %lf, %lf\r\n", icp_score, m_scene_align.m_pc_reg.m_inlier_threshold );
                screen_printf( "%s\r\n", m_scene_align.m_pc_reg.m_final_opt_summary.BriefReport().c_str() );

                m_logger_loop_closure.printf( "===============================================\r\n" );
                m_logger_loop_closure.printf( "%s -- %s\r\n", m_filename_vec[ keyframe_vec.size() - 1 ].c_str(), m_filename_vec[ his ].c_str() );
                m_logger_loop_closure.printf( "ICP inlier threshold = %lf, %lf\r\n", icp_score, m_scene_align.m_pc_reg.m_inlier_threshold );
                m_logger_loop_closure.printf( "%s\r\n", m_scene_align.m_pc_reg.m_final_opt_summary.BriefReport().c_str() );

                if ( m_scene_align.m_pc_reg.m_inlier_threshold > m_loop_closure_map_alignment_inlier_threshold * 2 )
                {
                    his += 10;
                    continue;
                }

                if ( m_scene_align.m_pc_reg.m_inlier_threshold < m_loop_closure_map_alignment_inlier_threshold )
                {
                    printf( "I believe this is true loop.\r\n" );
                    m_logger_loop_closure.printf( "I believe this is true loop.\r\n" );
                    auto Q_a = pose3d_vec[ his ].q;
                    auto Q_b = pose3d_vec[ pose3d_vec.size() - 1 ].q;
                    auto T_a = pose3d_vec[ his ].p;
                    auto T_b = pose3d_vec[ pose3d_vec.size() - 1 ].p;
                    auto ICP_q = m_scene_align.m_pc_reg.m_q_w_curr;
                    auto ICP_t = m_scene_align.m_pc_reg.m_t_w_curr;

                    ICP_t = ( ICP_q.inverse() * ( -ICP_t ) );
                    ICP_q = ICP_q.inverse();

                    screen_out << "ICP_q = " << ICP_q.coeffs().transpose() << std::endl;
                    screen_out << "ICP_t = " << ICP_t.transpose() << std::endl;
                    for ( int i = 0; i < 10; i++ )
                    {
                        screen_out << "-------------------------------------" << std::endl;
                        screen_out << ICP_q.coeffs().transpose() << std::endl;
                        screen_out << ICP_t.transpose() << std::endl;
                    }
                    Ceres_pose_graph_3d::VectorOfConstraints constrain_vec_temp;
                    constrain_vec_temp = constrain_vec;
                    constrain_vec_temp.push_back( Scene_alignment<float>::add_constrain_of_loop( pose3d_vec.size() - 1, his, Q_a, T_a, Q_b, T_b, ICP_q, ICP_t ) );
                    std::string path_name = m_loop_save_dir_name;
                    std::string g2o_filename = std::string( path_name ).append( "/loop.g2o" );
                    pose3d_map_ori = pose3d_map;
                    auto temp_pose_3d_map = pose3d_map;
                    Scene_alignment<float>::save_edge_and_vertex_to_g2o( g2o_filename.c_str(), temp_pose_3d_map, constrain_vec_temp );
                    Ceres_pose_graph_3d::pose_graph_optimization( temp_pose_3d_map, constrain_vec_temp );
                    Ceres_pose_graph_3d::out_put_poses( std::string( path_name ).append( "/poses_ori.txt" ), pose3d_map_ori );   // Save pose before optimized
                    Ceres_pose_graph_3d::out_put_poses( std::string( path_name ).append( "/poses_opm.txt" ), temp_pose_3d_map ); // Output poses after optimized
                    ///////////////////////////
                    // ANCHOR After loop
                    // 1. Lock the buffer
                    // 2. Update pose and map

                    int lidar_id = 0;
                    m_mul_lidar_management.m_mutex_lidar_mapping[lidar_id]->lock();
                    Eigen::Quaterniond q_diff_opm = ( pose3d_map_ori.find( pose3d_vec.size() - 1 )->second.q.inverse() ) * ( temp_pose_3d_map.find( pose3d_vec.size() - 1 )->second.q );
                    Eigen::Vector3d    t_diff_opm = temp_pose_3d_map.find( pose3d_vec.size() - 1 )->second.p - pose3d_map_ori.find( pose3d_vec.size() - 1 )->second.p;
                    
                    
                    loop_closure_update_buffer_for_matching(lidar_id , m_lidar_agent[ lidar_id ].m_q_w_curr * q_diff_opm, m_lidar_agent[ lidar_id ].m_t_w_last + t_diff_opm );

                    m_lidar_agent[ lidar_id ].m_q_w_curr = m_lidar_agent[ lidar_id ].m_q_w_curr * q_diff_opm;
                    m_lidar_agent[ lidar_id ].m_t_w_curr = m_lidar_agent[ lidar_id ].m_t_w_last + t_diff_opm;
                    publish_current_odometry( lidar_id );
                    m_mul_lidar_management.m_mutex_lidar_mapping[lidar_id]->unlock();

                    m_scene_align.dump_file_name( std::string( path_name ).append( "/file_name.txt" ), map_file_name );

                    loop_closure_pub_optimzed_path( lidar_id, temp_pose_3d_map );

                    for ( int pc_idx = ( int ) map_id_pc.size() - 1; pc_idx >= 0; pc_idx -= 1 )
                    {
                        screen_out << "*** Refine pointcloud, curren idx = " << pc_idx << " ***" << endl;
                        auto refined_pt = map_rfn.refine_pointcloud( map_id_pc, pose3d_map_ori, temp_pose_3d_map, pc_idx, 0 );
                        pcl::toROSMsg( refined_pt, ros_laser_cloud_surround );
                        ros_laser_cloud_surround.header.stamp = ros::Time::now();
                        ros_laser_cloud_surround.header.frame_id = "/camera_init";
                        m_pub_pc_aft_loop.publish( ros_laser_cloud_surround );
                        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                    }

                    //map_rfn.refine_mapping( path_name, 0 );
                    if ( 0 )
                    {
                        map_rfn.refine_mapping( map_id_pc, pose3d_map_ori, temp_pose_3d_map, 1 );
                        pcl::toROSMsg( map_rfn.m_pts_aft_refind, ros_laser_cloud_surround );
                        ros_laser_cloud_surround.header.stamp = ros::Time::now();
                        ros_laser_cloud_surround.header.frame_id = "/camera_init";
                        m_pub_pc_aft_loop.publish( ros_laser_cloud_surround );
                    }
                    if_end = 1;
                    break;
                }
                else
                {
                    his += 5;
                }
                if ( if_end )
                {
                    break;
                }
            }
            if ( if_end )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 500 ) );
                break;
            }
        }

        screen_out << m_timer.toc_string( "Find loop" ) << std::endl;

        m_scene_align.dump_file_name( std::string( m_loop_save_dir_name ).append( "/file_name.txt" ), map_file_name );

        if ( 1 )
        {

            m_timer.tic( "Pub surround pts" );
            pcl::toROSMsg( pt_full, ros_laser_cloud_surround );
            ros_laser_cloud_surround.header.stamp = ros::Time::now();
            ros_laser_cloud_surround.header.frame_id = "/camera_init";
            m_pub_debug_pts.publish( ros_laser_cloud_surround );
            screen_out << m_timer.toc_string( "Pub surround pts" ) << std::endl;
        }

        if ( if_end )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 500 ) );
            std::cout << "--------------------" << std::endl;
            std::cout << "Exit loop detection" << std::endl;

            break;
        }
    }
};

void Laser_mapping::service_pub_surround_pts()
{
    pcl::VoxelGrid<PointType> down_sample_filter_surface;
    down_sample_filter_surface.setLeafSize( m_surround_pointcloud_resolution, m_surround_pointcloud_resolution, m_surround_pointcloud_resolution );
    pcl::PointCloud<PointType> pc_temp;
    sensor_msgs::PointCloud2   ros_laser_cloud_surround;
    std::this_thread::sleep_for( std::chrono::nanoseconds( 10 ) );
    pcl::PointCloud<PointType>::Ptr laser_cloud_surround( new pcl::PointCloud<PointType>() );
    laser_cloud_surround->reserve( 1e8 );
    std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
    int last_update_index = 0;
    while ( 1 )
    {
        while ( m_current_frame_index - last_update_index < 100 )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 1500 ) );
        }
        last_update_index = m_current_frame_index;
        pcl::PointCloud<PointType> pc_temp;
        laser_cloud_surround->clear();
        if ( m_pt_cell_map_full.get_cells_size() == 0 )
            continue;

        int lidar_id = 0;
        std::vector<Points_cloud_map<float>::Mapping_cell_ptr> cell_vec = m_pt_cell_map_full.find_cells_in_radius( m_lidar_agent[ lidar_id ].m_t_w_curr, 1000.0 );
        for ( size_t i = 0; i < cell_vec.size(); i++ )
        {
            if ( m_down_sample_replace )
            {
                down_sample_filter_surface.setInputCloud( cell_vec[ i ]->get_pointcloud().makeShared() );
                down_sample_filter_surface.filter( pc_temp );
                if ( m_loop_closure_if_enable == false )
                    cell_vec[ i ]->set_pointcloud( pc_temp );
                *laser_cloud_surround += pc_temp;
            }
            else
            {
                *laser_cloud_surround += cell_vec[ i ]->get_pointcloud();
            }
        }
        if ( laser_cloud_surround->points.size() )
        {
            down_sample_filter_surface.setInputCloud( laser_cloud_surround );
            down_sample_filter_surface.filter( *laser_cloud_surround );
            pcl::toROSMsg( *laser_cloud_surround, ros_laser_cloud_surround );
            ros_laser_cloud_surround.header.stamp = ros::Time::now();
            ros_laser_cloud_surround.header.frame_id = "/camera_init";
            m_pub_laser_cloud_surround.publish( ros_laser_cloud_surround );
        }
        //screen_out << "~~~~~~~~~~~ " << "pub_surround_service, size = " << laser_cloud_surround->points.size()  << " ~~~~~~~~~~~" << endl;
    }
};

Eigen::Matrix<double, 3, 1> Laser_mapping::pcl_pt_to_eigend( PointType &pt )
{
    return Eigen::Matrix<double, 3, 1>( pt.x, pt.y, pt.z );
};

void Laser_mapping::find_min_max_intensity( const pcl::PointCloud<PointType>::Ptr pc_ptr, float &min_I, float &max_I )
{
    int pt_size = pc_ptr->size();
    min_I = 10000;
    max_I = -min_I;
    for ( int i = 0; i < pt_size; i++ )
    {
        min_I = std::min( pc_ptr->points[ i ].intensity, min_I );
        max_I = std::max( pc_ptr->points[ i ].intensity, max_I );
    }
};

float Laser_mapping::refine_blur( float in_blur, const float &min_blur, const float &max_blur )
{
    return ( in_blur - min_blur ) / ( max_blur - min_blur );
}

float Laser_mapping::compute_fov_angle( const PointType &pt )
{
    float sq_xy = sqrt( std::pow( pt.y / pt.x, 2 ) + std::pow( pt.z / pt.x, 2 ) );
    return atan( sq_xy ) * 57.3;
};


int Laser_mapping::if_matchbuff_and_pc_sync( float point_cloud_current_timestamp )
{
    if ( m_lastest_pc_matching_refresh_time < 0 )
        return 1;
    if ( point_cloud_current_timestamp - m_lastest_pc_matching_refresh_time < m_maximum_pointcloud_delay_time )
        return 1;
    if ( m_lastest_pc_reg_time == m_lastest_pc_matching_refresh_time ) // All is processed
        return 1;
    screen_printf( "*** Current pointcloud timestamp = %.3f, lastest buff timestamp = %.3f, lastest_pc_reg_time = %.3f ***\r\n",
                   point_cloud_current_timestamp,
                   m_lastest_pc_matching_refresh_time,
                   m_lastest_pc_reg_time );
    //cout << "~~~~~~~~~~~~~~~~ Wait sync, " << point_cloud_current_timestamp << ", " << m_lastest_pc_matching_refresh_time << endl;

    return 0;
};



// ANCHOR process()
void Laser_mapping::process()
{
    double first_time_stamp = -1;
    m_last_max_blur = 0.0;

    m_service_pub_surround_pts = new std::future<void>( std::async( std::launch::async, &Laser_mapping::service_pub_surround_pts, this ) );
    if ( m_loop_closure_if_enable )
    {
        m_service_loop_detection = new std::future<void>( std::async( std::launch::async, &Laser_mapping::service_loop_detection, this ) );
    }
    timer_all.tic();
    while ( 1 )
    {

        m_logger_common.printf( "------------------\r\n" );
        while ( m_queue_avail_data.empty() )
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        m_mutex_buf.lock();
        while ( m_queue_avail_data.size() >= ( unsigned int ) m_max_buffer_size )
        {
            ROS_WARN( "Drop lidar frame in mapping for real time performance !!!" );
            ( *m_logger_common.get_ostream() ) << "Drop lidar frame in mapping for real time performance !!!" << endl;
            m_queue_avail_data.pop();
        }

        std::shared_ptr<Data_pair> current_data_pair =  m_queue_avail_data.front();
        m_queue_avail_data.pop();
        m_mutex_buf.unlock();
        
        m_timer.tic( "Prepare to enter thread" );
        m_time_pc_corner_past = current_data_pair->m_pc_corner.header.stamp.toSec();
        if ( first_time_stamp < 0 )
        {
            first_time_stamp = m_time_pc_corner_past;
        }
        ( *m_logger_common.get_ostream() ) << "Messgage time stamp = " << m_time_pc_corner_past - first_time_stamp << endl;

        if ( m_if_multiple_lidar )
        {
            if ( m_mul_lidar_management.registration_lidar_scans( current_data_pair ) )
            {
                Common_tools::maintain_maximum_thread_pool<std::future<int> *>( m_thread_pool, m_maximum_parallel_thread );
                if ( 1 )
                {
                    std::future<int> *thd = new std::future<int>( std::async( std::launch::async, &Laser_mapping::process_new_scan, this, m_mul_lidar_management.get_data_pair_for_registraction( current_data_pair ) ) );
                    m_thread_pool.push_back( thd );
                }
                else
                {
                    process_new_scan( m_mul_lidar_management.get_data_pair_for_registraction( current_data_pair ) );
                }
            }
        }
        else
        {
            Common_tools::maintain_maximum_thread_pool<std::future<int> *>( m_thread_pool, m_maximum_parallel_thread );
            std::future<int> *thd = new std::future<int>( std::async( std::launch::async, &Laser_mapping::process_new_scan, this, current_data_pair ) );
            m_thread_pool.push_back( thd );
        }

        // thd->get();
        // *( m_logger_timer.get_ostream() ) << m_timer.toc_string( "Prepare to enter thread" ) << std::endl;

        std::this_thread::sleep_for( std::chrono::nanoseconds( 10 ) );
    }
}