#include "laser_mapping.hpp"

extern int    g_if_undistore;
extern double history_add_t_step;
extern double history_add_angle_step;

extern int if_motion_deblur;
int        g_receive_count = 0;
int        g_lidar_frame_idx = 0;

int if_degrade(vec_6 eig_vec)
{
    g_lidar_frame_idx++;
    if((eig_vec(0) / eig_vec(2) <= 0.1) && (g_lidar_frame_idx) > 100 )
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void Laser_mapping::set_EFK_extrinsic(int id )
{
    // cout << "set_EFK_extrinsic" << endl;
    m_global_ekf.reset_extrinsic(id);
}

// ANCHOR map fusion
void Laser_mapping::map_fusion(int lidar_id, int tar_id)
{
    if(lidar_id > 6 || lidar_id == 1 || lidar_id == 2)
        return;
    Eigen::Quaterniond q_pre;
        vec_3 t_pre;
    std::set<Points_cloud_map<float>::Mapping_cell_ptr> cell_over_lap_a, cell_over_lap_b;
    if((m_lidar_agent[lidar_id].m_trajectory_time.size() % 10)!= 0 ) // Limit the mapping frequency
        return;
    m_mul_lidar_management.find_overlap_of_two_map( m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ], m_mul_lidar_management.m_pt_cell_map_planes[ tar_id ],
                                                    cell_over_lap_a, cell_over_lap_b );
    
    float overlap_ratio = cell_over_lap_a.size() / (float) std::min( (m_mul_lidar_management.m_pt_cell_map_planes[ 0 ].m_cell_vec.size()) , 
            m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].m_map_pt_cell.size());

    if ( m_lidar_agent[ 0 ].m_t_w_curr.norm() > 15 )
    {
        m_mul_lidar_management.m_map_index[ lidar_id ] = tar_id;
        m_mul_lidar_management.m_if_have_merge[ tar_id ] = 1;
        m_mul_lidar_management.m_if_have_merge[ lidar_id ] = 1;

        m_mul_lidar_management.m_pt_cell_map_corners[ lidar_id ].clear_data();
        m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].clear_data();
        int ekf_lidar_idx = lidar_id;
        if ( ekf_lidar_idx > 2 )
        {
            ekf_lidar_idx -= 2;
        }

        set_EFK_extrinsic(ekf_lidar_idx);
        m_global_ekf.get_prediction_of_idx_lidar(ekf_lidar_idx, q_pre, t_pre);
        m_lidar_agent[ lidar_id ].m_q_w_curr = q_pre;
        m_lidar_agent[ lidar_id ].m_t_w_curr = t_pre;
        m_lidar_agent[ lidar_id ].m_q_w_last = m_lidar_agent[ lidar_id ].m_q_w_curr;
        m_lidar_agent[ lidar_id ].m_t_w_last = m_lidar_agent[ lidar_id ].m_t_w_curr;
        cout << "Lidar id " << lidar_id << " <--> 0 | " << overlap_ratio << " | " << m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_inlier_threshold << " ";
        cout << "Align fail, size too big, force merge!!!" << endl;
    }
    else
    {
        // return;
    }
    if ( cell_over_lap_a.size() > m_minimum_overlap_cells_num )
    {
        Maps_keyframe<float> keyframe_a, keyframe_b;
        keyframe_a.add_cells( cell_over_lap_a );
        keyframe_b.add_cells( cell_over_lap_b );
        m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_if_verbose_screen_printf = 1;
        m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_para_scene_alignments_maximum_residual_block = 1e4;
        m_mul_lidar_management.m_scene_align_vec[ lidar_id ].set_downsample_resolution( 0.2, 0.2 );
        m_mul_lidar_management.m_scene_align_vec[ lidar_id ].find_tranfrom_of_two_mappings( &keyframe_a, &keyframe_b, 0, " ", 1 );
        if ( m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_inlier_threshold < 0.20)
        //if(1)
        {
            if( m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_t_w_curr.norm()  > 0.8 )
            {
                return;
            }
            std::unique_lock<std::mutex> lock( *m_mul_lidar_management.m_mutex_lidar_mapping[ tar_id ] );
            m_mul_lidar_management.m_if_have_merge[ lidar_id ] = 1;
            cout << "Lidar id " << lidar_id << " <--> 0 | " << overlap_ratio << " | " << m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_inlier_threshold << " ";
            cout << "Align successful, need to merge two mapping" << endl;
            cout << std::setprecision( 3 ) << m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_q_w_curr.coeffs().transpose() << endl;
            cout << std::setprecision( 3 ) << m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_t_w_curr.transpose() << endl;
            pcl::PointCloud<PointType> plane_ori_pointcloud = m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].get_all_pointcloud();
            pcl::PointCloud<PointType> line_ori_pointcloud = m_mul_lidar_management.m_pt_cell_map_corners[ lidar_id ].get_all_pointcloud();

            plane_ori_pointcloud = PCL_TOOLS::pointcloud_transfrom<double, PointType>( plane_ori_pointcloud,
                                                                                       m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_q_w_curr.toRotationMatrix(),
                                                                                       m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_t_w_curr );

            line_ori_pointcloud = PCL_TOOLS::pointcloud_transfrom<double, PointType>( line_ori_pointcloud,
                                                                                      m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_q_w_curr.toRotationMatrix(),
                                                                                      m_mul_lidar_management.m_scene_align_vec[ lidar_id ].m_pc_reg.m_t_w_curr );
            m_mul_lidar_management.m_pt_cell_map_corners[ lidar_id ].clear_data();
            m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].save_to_file( TEMP_LOG_SAVE_DIR, "0_a.json" );
            m_mul_lidar_management.m_pt_cell_map_corners[ tar_id ].append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, pcl::PointXYZI>( line_ori_pointcloud.makeShared() ) );

            m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].clear_data();
            m_mul_lidar_management.m_pt_cell_map_planes[ tar_id ].append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, pcl::PointXYZI>( line_ori_pointcloud.makeShared() ) );
            m_mul_lidar_management.m_pt_cell_map_planes[ tar_id ].save_to_file( TEMP_LOG_SAVE_DIR, "0_c.json" );
            m_mul_lidar_management.m_map_index[ lidar_id ] = tar_id;
            m_mul_lidar_management.m_if_have_merge[tar_id] = 1;

        }
        
    }
}

//ANCHOR update_buff_for_matching
void Laser_mapping::update_buff_for_matching( int lidar_id )
{
    // if ( m_lastest_pc_matching_refresh_time == m_lastest_pc_reg_time )
    // return;

    m_timer.tic( "Update buff for matching" );
    pcl::VoxelGrid<PointType> down_sample_filter_corner = m_down_sample_filter_corner;
    pcl::VoxelGrid<PointType> down_sample_filter_surface = m_down_sample_filter_surface;
    down_sample_filter_corner.setLeafSize( m_line_resolution * m_mapping_feature_downsample_scale, m_line_resolution * m_mapping_feature_downsample_scale, m_line_resolution * m_mapping_feature_downsample_scale );
    down_sample_filter_surface.setLeafSize( m_plane_resolution * m_mapping_feature_downsample_scale, m_plane_resolution * m_mapping_feature_downsample_scale, m_plane_resolution * m_mapping_feature_downsample_scale );

    pcl::PointCloud<PointType> laser_cloud_corner_from_map, laser_cloud_surf_from_map;

    laser_cloud_corner_from_map.reserve( 1e8 );
    laser_cloud_surf_from_map.reserve( 1e8 );

    laser_cloud_corner_from_map.clear();
    laser_cloud_surf_from_map.clear();

    int hack_select_id = lidar_id;
    if ( m_matching_mode ) // match with global mappings
    {
        pcl::VoxelGrid<PointType> down_sample_filter_corner = m_down_sample_filter_corner;
        pcl::VoxelGrid<PointType> down_sample_filter_surface = m_down_sample_filter_surface;
        if ( m_mul_lidar_management.m_if_have_merge[hack_select_id] == 1 )
            lidar_id = 0;
        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->lock();
        if(lidar_id != 0 && m_mul_lidar_management.m_if_have_merge[hack_select_id] == 0 )
        {
            map_fusion(lidar_id, 0);
        }
       
        std::vector<Points_cloud_map<float>::Mapping_cell_ptr> corner_cell_vec = m_mul_lidar_management.m_pt_cell_map_corners[ lidar_id ].find_cells_in_radius( m_lidar_agent[ lidar_id ].m_t_w_curr, m_maximum_search_range_corner );
        std::vector<Points_cloud_map<float>::Mapping_cell_ptr> plane_cell_vec = m_mul_lidar_management.m_pt_cell_map_planes[ lidar_id ].find_cells_in_radius( m_lidar_agent[ lidar_id ].m_t_w_curr, m_maximum_search_range_surface );

        int                        corner_cell_numbers_in_fov = 0;
        int                        surface_cell_numbers_in_fov = 0;
        pcl::PointCloud<PointType> pc_temp;

        for ( size_t i = 0; i < corner_cell_vec.size(); i++ )
        {
            int if_in_fov = if_pt_in_fov( corner_cell_vec[ i ]->m_center.cast<double>() );
            if ( if_in_fov == 0 )
            {
                continue;
            }
            corner_cell_numbers_in_fov++;
            if ( m_down_sample_replace )
            {
                // down_sample_filter_corner.setInputCloud( corner_cell_vec[ i ]->get_pointcloud().makeShared() );
                down_sample_filter_corner.setInputCloud( corner_cell_vec[ i ]->get_oldest_pointcloud().makeShared() );
                down_sample_filter_corner.filter( pc_temp );
                corner_cell_vec[ i ]->set_pointcloud( pc_temp );
            }
            else
            {
                pc_temp = corner_cell_vec[ i ]->get_pointcloud();
            }
            laser_cloud_corner_from_map += pc_temp;
        }

        for ( size_t i = 0; i < plane_cell_vec.size(); i++ )
        {
            int if_in_fov = if_pt_in_fov( plane_cell_vec[ i ]->m_center.cast<double>() );
            if ( if_in_fov == 0 )
            {
                continue;
            }
            surface_cell_numbers_in_fov++;

            if ( m_down_sample_replace )
            {
                // down_sample_filter_surface.setInputCloud( plane_cell_vec[ i ]->get_pointcloud().makeShared() );
                down_sample_filter_surface.setInputCloud( plane_cell_vec[ i ]->get_oldest_pointcloud().makeShared() );
                down_sample_filter_surface.filter( pc_temp );
                plane_cell_vec[ i ]->set_pointcloud( pc_temp );
            }
            else
            {
                pc_temp = plane_cell_vec[ i ]->get_pointcloud();
            }
            laser_cloud_surf_from_map += pc_temp;
        }
        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->unlock();
    }
    else // match with local mappings
    {
        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->lock();
        for ( auto it = m_mul_lidar_management.m_laser_cloud_corner_history_vec[ lidar_id ].begin(); it != m_mul_lidar_management.m_laser_cloud_corner_history_vec[ lidar_id ].end(); it++ )
        {
            laser_cloud_corner_from_map += ( *it );
        }

        for ( auto it = m_mul_lidar_management.m_laser_cloud_surface_history_vec[ lidar_id ].begin(); it != m_mul_lidar_management.m_laser_cloud_surface_history_vec[ lidar_id ].end(); it++ )
        {
            laser_cloud_surf_from_map += ( *it );
        }

        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->unlock();
    }

    if ( laser_cloud_surf_from_map.points.size() == 0 && laser_cloud_corner_from_map.points.size() == 0 )
    {
        return;
    }

    if ( 0 )
    {
        // pcl::io::savePCDFile(string( "surface_for_matching_" ).append( std::to_string( lidar_id )).append(".pcd"), laser_cloud_surf_from_map );
        // pcl::io::savePCDFile(string( "corner_for_matching_" ).append( std::to_string( lidar_id )).append(".pcd"), laser_cloud_corner_from_map );
        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->lock();
        if ( m_mul_lidar_management.m_lidar_frame_count[ lidar_id ] % 10 == 0 && laser_cloud_surf_from_map.points.size() && laser_cloud_corner_from_map.points.size() )
        {
            m_pcl_tools_raw.save_to_pcd_files( string( "surface_for_matching_" ).append( std::to_string( lidar_id ) ), laser_cloud_surf_from_map, m_mul_lidar_management.m_lidar_frame_count[ lidar_id ], 0 );
            m_pcl_tools_raw.save_to_pcd_files( string( "corner_for_matching_" ).append( std::to_string( lidar_id ) ), laser_cloud_corner_from_map, m_mul_lidar_management.m_lidar_frame_count[ lidar_id ], 0 );
        }
        m_mul_lidar_management.m_lidar_frame_count[ lidar_id ]++;
        m_mul_lidar_management.m_mutex_lidar_mapping[ lidar_id ]->unlock();
    }
    // cout << "Matching, pointcloud id = " << lidar_id << " , size = " << laser_cloud_surf_from_map.points.size() << " , " << laser_cloud_corner_from_map.points.size() << endl;

    // Create the filtering object

    down_sample_filter_corner.setInputCloud( laser_cloud_corner_from_map.makeShared() );
    down_sample_filter_corner.filter( laser_cloud_corner_from_map );

    down_sample_filter_surface.setInputCloud( laser_cloud_surf_from_map.makeShared() );
    down_sample_filter_surface.filter( laser_cloud_surf_from_map );


    pcl::KdTreeFLANN<PointType> kdtree_corner_from_map;
    pcl::KdTreeFLANN<PointType> kdtree_surf_from_map;

    if ( laser_cloud_corner_from_map.points.size() && laser_cloud_surf_from_map.points.size() )
    {
        kdtree_corner_from_map.setInputCloud( laser_cloud_corner_from_map.makeShared() );
        kdtree_surf_from_map.setInputCloud( laser_cloud_surf_from_map.makeShared() );
    }

    m_if_mapping_updated_corner = false;
    m_if_mapping_updated_surface = false;

    m_mul_lidar_management.m_mutex_buff_for_matching_corner[ hack_select_id ]->lock();
    m_mul_lidar_management.m_laser_cloud_corner_from_map_last_vec[ hack_select_id ] = laser_cloud_corner_from_map;
    m_mul_lidar_management.m_kdtree_corner_from_map_last[ hack_select_id ] = kdtree_corner_from_map;
    m_mul_lidar_management.m_mutex_buff_for_matching_corner[ hack_select_id ]->unlock();

    m_mul_lidar_management.m_mutex_buff_for_matching_surface[ hack_select_id ]->lock();
    m_mul_lidar_management.m_laser_cloud_surf_from_map_last_vec[ hack_select_id ] = laser_cloud_surf_from_map;
    m_mul_lidar_management.m_kdtree_surf_from_map_last[ hack_select_id ] = kdtree_surf_from_map;
    m_mul_lidar_management.m_mutex_buff_for_matching_surface[ hack_select_id ]->unlock();

    if ( ( m_lastest_pc_reg_time > m_lastest_pc_matching_refresh_time ) || ( m_lastest_pc_reg_time < 10 ) )
    {
        m_lastest_pc_matching_refresh_time = m_lastest_pc_reg_time;
    }
    *m_logger_matching_buff.get_ostream() << m_timer.toc_string( "Update buff for matching" ) << std::endl;

    if ( IF_PUBLISH_MATHCING_BUFFER & ( !( IF_PUBLISH_SURFACE_AND_CORNER_PTS ) ) )
    {
        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg( laser_cloud_surf_from_map, laserCloudMsg );
        laserCloudMsg.header.stamp = ros::Time().now();
        laserCloudMsg.header.frame_id = "/camera_init";
        m_pub_match_surface_pts.publish( laserCloudMsg );
    }
}

double Laser_mapping::get_average_time(Loam_livox_custom_point_cloud & full_point_cloud_msg)
{
    int pt_size = full_point_cloud_msg.m_point_size;
    double res_time = 0;
    for (size_t i = 0; i< pt_size; i++)
    {
        res_time = res_time + full_point_cloud_msg.m_timestamp[i] / (double) pt_size;
    }
    return res_time;
}

int check_if_reset(int lidar_index, Global_ekf & m_global_ekf)
{
    int res = 0;

    vec_3 t_vec_0 = m_global_ekf.m_full_state.block( 15, 0, 3, 1 );
    vec_3 t_vec = m_global_ekf.m_full_state.block( 15 + 6 * lidar_index, 0, 3, 1 ) - t_vec_0;
    // if ( lidar_index == 2 )
    // {
    //     if ( t_vec( 1 ) < -0.0 )
    //         res = 1;
    // }
    // if ( lidar_index == 1 )
    // {

    //     if ( t_vec( 1 ) > 0.0 )
    //         res = 1;
    // }
    // if(abs(t_vec(1)) > 0.4)
    //     res = 1;

    if ( t_vec.norm() > 0.5 )
        res = 1;
    if ( res )
    {
        cout << "|-> "<< lidar_index << " | " << t_vec.transpose() << " | " << t_vec_0.transpose() << endl;
    }
    return res;
}

// ANCHOR process new scan()
int Laser_mapping::process_new_scan( std::shared_ptr<Data_pair> current_data_pair )
{
    m_timer.tic( "Frame process" );
    m_timer.tic( "Query points for match" );

    Common_tools::Timer timer_frame;
    timer_frame.tic();

    Loam_livox_custom_point_cloud full_point_cloud_msg;
    pcl::PointCloud<PointType>    current_laser_cloud_full, current_laser_cloud_full_intensity, current_laser_cloud_corner_last, current_laser_cloud_surf_last;

    pcl::VoxelGrid<PointType>   down_sample_filter_corner = m_down_sample_filter_corner;
    pcl::VoxelGrid<PointType>   down_sample_filter_surface = m_down_sample_filter_surface;
    pcl::KdTreeFLANN<PointType> kdtree_corner_from_map;
    pcl::KdTreeFLANN<PointType> kdtree_surf_from_map;

    m_mutex_querypointcloud.lock();
    full_point_cloud_msg = current_data_pair->m_pc_full;
    Custom_point_cloud_interface::msg_to_pcl_pc( current_data_pair->m_pc_corner, current_laser_cloud_corner_last );
    Custom_point_cloud_interface::msg_to_pcl_pc( current_data_pair->m_pc_plane, current_laser_cloud_surf_last );
    Custom_point_cloud_interface::msg_to_pcl_pc( current_data_pair->m_pc_full, current_laser_cloud_full );
    int lidar_id = full_point_cloud_msg.m_lidar_id;
    //cout << "Current lidar id = " << lidar_id << endl;
    int raw_lidar_id = lidar_id;
    int ekf_lidar_idx = raw_lidar_id;
    if(ekf_lidar_idx > 2)
    {
        ekf_lidar_idx -= 2;
    }

    memcpy( m_lidar_agent[ lidar_id ].m_odometry_RT_curr, current_data_pair->m_buffer_RT, 7 * sizeof( double ) );

    if ( m_if_load_extrinsic )
    {
        Eigen::Matrix<double, 4, 4> tr_mat_44;
        tr_mat_44.setIdentity();
        tr_mat_44.block( 0, 0, 3, 3 ) = m_mul_lidar_management.m_lidar_extrinsic_R_vec[ lidar_id ].toRotationMatrix();
        tr_mat_44.block( 0, 3, 3, 1 ) = m_mul_lidar_management.m_lidar_extrinsic_T_vec[ lidar_id ];
        pcl::transformPointCloud( current_laser_cloud_corner_last, current_laser_cloud_corner_last, tr_mat_44 );
        pcl::transformPointCloud( current_laser_cloud_surf_last, current_laser_cloud_surf_last, tr_mat_44 );
        pcl::transformPointCloud( current_laser_cloud_full, current_laser_cloud_full, tr_mat_44 );
    }
    current_laser_cloud_full_intensity = current_laser_cloud_full;
    for ( int i = 0; i < current_laser_cloud_full_intensity.points.size(); i++ )
    {
        current_laser_cloud_full_intensity.points[ i ].intensity = full_point_cloud_msg.m_intensity[ i ];
    }

    int matching_pointcloud_id = m_mul_lidar_management.m_map_index[ lidar_id ];

    m_mutex_querypointcloud.unlock();
    if ( m_para_if_force_update_buffer_for_matching )
    {
        update_buff_for_matching(matching_pointcloud_id);
    }
    double                         rt_from_odom[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    Eigen::Map<Eigen::Quaterniond> incre_r_from_odometry = Eigen::Map<Eigen::Quaterniond>( rt_from_odom );
    Eigen::Map<Eigen::Vector3d>    incre_t_from_odometry = Eigen::Map<Eigen::Vector3d>( rt_from_odom + 4 );


    //delete current_data_pair;
    float min_t, max_t;
    find_min_max_intensity( current_laser_cloud_full.makeShared(), min_t, max_t );

    double point_cloud_current_timestamp = min_t;

    if ( point_cloud_current_timestamp > m_lastest_pc_income_time )
    {
        m_lastest_pc_income_time = point_cloud_current_timestamp;
    }
    
    point_cloud_current_timestamp = get_average_time( full_point_cloud_msg );
    m_time_odom = point_cloud_current_timestamp;
    m_minimum_pt_time_stamp = m_last_time_stamp;
    m_maximum_pt_time_stamp = max_t;
    m_last_time_stamp = max_t;
    Point_cloud_registration pc_reg;
    init_pointcloud_registration( pc_reg, matching_pointcloud_id );
    m_current_frame_index++;

    double time_odom = point_cloud_current_timestamp;
    screen_printf( "****** Before timestamp info = [%.6f, %.6f, %.6f, %.6f ] ****** \r\n", m_minimum_pt_time_stamp, m_maximum_pt_time_stamp, min_t, m_lastest_pc_matching_refresh_time );

    m_timer.tic( "Wait sync" );
    while ( !if_matchbuff_and_pc_sync( point_cloud_current_timestamp ) )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    *( m_logger_timer.get_ostream() ) << m_timer.toc_string( "Wait sync" ) << std::endl;
    screen_printf( "****** After timestamp info = [%.6f, %.6f, %.6f, %.6f ] ****** \r\n", m_minimum_pt_time_stamp, m_maximum_pt_time_stamp, min_t, m_lastest_pc_matching_refresh_time );

    pcl::PointCloud<PointType>::Ptr laserCloudCornerStack( new pcl::PointCloud<PointType>() );
    pcl::PointCloud<PointType>::Ptr laserCloudSurfStack( new pcl::PointCloud<PointType>() );

    if ( m_if_input_downsample_mode )
    {
        down_sample_filter_corner.setInputCloud( current_laser_cloud_corner_last.makeShared() );
        down_sample_filter_corner.filter( *laserCloudCornerStack );
        down_sample_filter_surface.setInputCloud( current_laser_cloud_surf_last.makeShared() );
        down_sample_filter_surface.filter( *laserCloudSurfStack );
    }
    else
    {
        *laserCloudCornerStack = current_laser_cloud_corner_last;
        *laserCloudSurfStack = current_laser_cloud_surf_last;
    }

    int laser_corner_pt_num = laserCloudCornerStack->points.size();
    int laser_surface_pt_num = laserCloudSurfStack->points.size();

    if ( m_if_save_to_pcd_files && PCD_SAVE_RAW )
    {
        m_pcl_tools_raw.save_to_pcd_files( "raw", current_laser_cloud_full, m_current_frame_index );
    }

    m_lidar_agent[ lidar_id ].m_q_w_last = m_lidar_agent[ lidar_id ].m_q_w_curr;
    m_lidar_agent[ lidar_id ].m_t_w_last = m_lidar_agent[ lidar_id ].m_t_w_curr;

    pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map( new pcl::PointCloud<PointType>() );
    pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map( new pcl::PointCloud<PointType>() );
    int                             reg_res = 0;
    
    
    m_mul_lidar_management.m_mutex_buff_for_matching_corner[matching_pointcloud_id]->lock();
    *laser_cloud_corner_from_map = (m_mul_lidar_management.m_laser_cloud_corner_from_map_last_vec[matching_pointcloud_id]);
    kdtree_corner_from_map = m_mul_lidar_management.m_kdtree_corner_from_map_last[matching_pointcloud_id];
    m_mul_lidar_management.m_mutex_buff_for_matching_corner[matching_pointcloud_id]->unlock();

    m_mul_lidar_management.m_mutex_buff_for_matching_surface[matching_pointcloud_id]->lock();
    *laser_cloud_surf_from_map = (m_mul_lidar_management.m_laser_cloud_surf_from_map_last_vec[matching_pointcloud_id]);
    kdtree_surf_from_map = m_mul_lidar_management.m_kdtree_surf_from_map_last[matching_pointcloud_id];
    m_mul_lidar_management.m_mutex_buff_for_matching_surface[matching_pointcloud_id]->unlock();

    Eigen::Quaterniond q_pre;
    vec_3              t_pre;

    // ANCHOR EKF prediction
    if(1)
    {
        // ANCHOR force update
        if ( m_lidar_agent[ 0 ].m_t_w_curr.norm() < 30 )
            m_para_if_force_update_buffer_for_matching = 1;
        else
            m_para_if_force_update_buffer_for_matching = 0;

        if(1)
        {
            double             angle_dis;
            double             t_dis;
            if(check_if_reset(ekf_lidar_idx, m_global_ekf)  )
            {
                set_EFK_extrinsic(ekf_lidar_idx);
            }
            m_global_ekf.prediction( ekf_lidar_idx, 0, point_cloud_current_timestamp);
            m_global_ekf.get_prediction_of_idx_lidar( ekf_lidar_idx, q_pre, t_pre );

            angle_dis = q_pre.angularDistance( pc_reg.m_q_w_curr )*57.3;
            t_dis  = (t_pre- pc_reg.m_t_w_curr).norm();
            
            if ( (angle_dis < 20)  && (t_dis < 5.0) )
            {   
                if(lidar_id != 0 && m_mul_lidar_management.m_if_have_merge[ lidar_id ])
                {
                    // pc_reg.m_para_icp_max_iterations  = 10;
                }
                pc_reg.m_inlier_ratio = 0.8;
                pc_reg.m_inlier_threshold = 0.02;
                pc_reg.m_q_w_curr = q_pre;
                pc_reg.m_q_w_last = q_pre;
                pc_reg.m_t_w_curr = t_pre;
                pc_reg.m_t_w_last = t_pre;
            }
            else
            {
                q_pre = pc_reg.m_q_w_curr;
                t_pre = pc_reg.m_t_w_curr;
            }
        }
    }

    memcpy( pc_reg.m_para_buffer_incremental, m_lidar_agent[ matching_pointcloud_id ].m_buffer_RT_last_incre, 7 * sizeof( double ) );

    // ANCHOR Find out incremental transfrom
    reg_res = pc_reg.find_out_incremental_transfrom( laser_cloud_corner_from_map, laser_cloud_surf_from_map,
                                                     kdtree_corner_from_map, kdtree_surf_from_map,
                                                     laserCloudCornerStack, laserCloudSurfStack );
    
    int if_degrage = if_degrade( pc_reg.m_cov_mat_eig_vec );
    if(if_degrage && 0)
    {
        
        printf( "=================== degrage id = %d ===================\r\n", raw_lidar_id );
        pc_reg.m_t_w_incre.setZero();
        pc_reg.m_q_w_incre.setIdentity();
        pc_reg.update_transform();
        //return 0;
    }
    if ( reg_res == 0 )
    {
        return 0;
    }
    m_timer.tic( "Add new frame" );
    
    // ANCHOR EKF Mes update
    if(1)
    {
        // ANCHOR  Update_current_pose_of_lidar
        screen_printf( "Current time is %f\r\n", point_cloud_current_timestamp );
        int save_log=0;
        if(lidar_id == 0 || m_mul_lidar_management.m_if_have_merge[lidar_id])
        {
            save_log = 1;
        }
        
        m_lidar_agent[ lidar_id ].update( point_cloud_current_timestamp, pc_reg.m_q_w_curr, pc_reg.m_t_w_curr, pc_reg.m_cov_mat, save_log );

        double obs_gain = 10.0;
        if(lidar_id == 0)
        {
            obs_gain = 1.0;
        }

        //if(save_log == 1)
        // ANCHOR Fix rt
        int if_fix_rt = 1;
        if(m_lidar_agent[ lidar_id ].m_t_w_curr.norm() > 10)
        {
            if_fix_rt = 0;
        }

        //if ( 1 )
        if ( ekf_lidar_idx == 0 )
        {
            m_global_ekf.prediction( ekf_lidar_idx, 0, point_cloud_current_timestamp );
            m_global_ekf.ekf_measureament_update( ekf_lidar_idx, m_lidar_agent[ lidar_id ].m_trajectory_time.size() - 1, point_cloud_current_timestamp, 1.0, if_fix_rt );
        }
        m_global_ekf.dump_to_log( m_lidar_agent[ lidar_id ].m_trajectory_time.size() );

        m_lastest_pc_reg_time = 0;
    }
    else
    {
        *( m_logger_common.get_ostream() ) << "***** older update, reject update pose *****" << endl;
    }

    PointType                       pointOri, pointSel;
    pcl::PointCloud<PointType>::Ptr pc_new_feature_corners( new pcl::PointCloud<PointType>() );
    pcl::PointCloud<PointType>::Ptr pc_new_feature_surface( new pcl::PointCloud<PointType>() );
    for ( int i = 0; i < laser_corner_pt_num; i++ )
    {
        pc_reg.pointAssociateToMap( &laserCloudCornerStack->points[ i ], &pointSel, refine_blur( laserCloudCornerStack->points[ i ].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp ), g_if_undistore );
        pc_new_feature_corners->push_back( pointSel );
    }

    for ( int i = 0; i < laser_surface_pt_num; i++ )
    {
        pc_reg.pointAssociateToMap( &laserCloudSurfStack->points[ i ], &pointSel, refine_blur( laserCloudSurfStack->points[ i ].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp ), g_if_undistore );
        pc_new_feature_surface->push_back( pointSel );
    }
    down_sample_filter_corner.setInputCloud( pc_new_feature_corners );
    down_sample_filter_corner.filter( *pc_new_feature_corners );
    down_sample_filter_surface.setInputCloud( pc_new_feature_surface );
    down_sample_filter_surface.filter( *pc_new_feature_surface );

    double r_diff = m_lidar_agent[ lidar_id ].m_q_w_curr.angularDistance( m_last_his_add_q ) * 570000.3;
    double t_diff = ( m_lidar_agent[ lidar_id ].m_t_w_curr - m_last_his_add_t ).norm()*100000;

    pc_reg.pointcloudAssociateToMap( current_laser_cloud_full, current_laser_cloud_full, g_if_undistore );

    m_mul_lidar_management.m_mutex_lidar_mapping[ matching_pointcloud_id ]->lock();

    if ( m_mul_lidar_management.m_laser_cloud_corner_history_vec[ lidar_id ].size() < ( size_t ) m_maximum_history_size ||
         ( t_diff > history_add_t_step ) ||
         ( r_diff > history_add_angle_step * 57.3 ) )
    {
        m_last_his_add_q = m_lidar_agent[ lidar_id ].m_q_w_curr;
        m_last_his_add_t = m_lidar_agent[ lidar_id ].m_t_w_curr;
        //if ( !if_degrage )

        // cout <<"Input id: "<< lidar_id <<" ,pushback point cloud to id "<< matching_pointcloud_id <<endl;
        m_mul_lidar_management.m_laser_cloud_corner_history_vec[ matching_pointcloud_id ].push_back( *pc_new_feature_corners );
        m_mul_lidar_management.m_laser_cloud_surface_history_vec[ matching_pointcloud_id ].push_back( *pc_new_feature_surface );

        m_mutex_dump_full_history.lock();
        m_laser_cloud_corner_for_keyframe.push_back( *pc_new_feature_corners );
        m_laser_cloud_surface_for_keyframe.push_back( *pc_new_feature_surface );
        m_laser_cloud_full_history.push_back( current_laser_cloud_full );
        m_his_reg_error.push_back( pc_reg.m_inlier_threshold );
        m_mutex_dump_full_history.unlock();
    }
    else
    {
        screen_printf( "==== Reject add history, T_norm = %.2f, R_norm = %.2f ====\r\n", t_diff, r_diff );
    }
    
    if ( m_mul_lidar_management.m_laser_cloud_corner_history_vec[matching_pointcloud_id].size() > ( size_t ) m_maximum_history_size )
    {
        ( m_mul_lidar_management.m_laser_cloud_corner_history_vec[matching_pointcloud_id].front() ).clear();
        m_mul_lidar_management.m_laser_cloud_corner_history_vec[matching_pointcloud_id].pop_front();
    }

    if ( m_mul_lidar_management.m_laser_cloud_surface_history_vec[matching_pointcloud_id].size() > ( size_t ) m_maximum_history_size )
    {
        ( m_mul_lidar_management.m_laser_cloud_surface_history_vec[matching_pointcloud_id].front() ).clear();
        m_mul_lidar_management.m_laser_cloud_surface_history_vec[matching_pointcloud_id].pop_front();
    }

    if ( m_laser_cloud_full_history.size() > ( size_t ) m_maximum_history_size )
    {
        m_mutex_dump_full_history.lock();

        m_laser_cloud_full_history.front().clear();
        m_laser_cloud_full_history.pop_front();

        m_laser_cloud_corner_for_keyframe.front().clear();
        m_laser_cloud_corner_for_keyframe.pop_front();

        m_laser_cloud_surface_for_keyframe.front().clear();
        m_laser_cloud_surface_for_keyframe.pop_front();

        m_his_reg_error.pop_front();
        m_mutex_dump_full_history.unlock();
    }
    m_if_mapping_updated_corner = true;
    m_if_mapping_updated_surface = true;

    if ( m_matching_mode != 0 )
    {
        m_mul_lidar_management.m_pt_cell_map_corners[ matching_pointcloud_id ].append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>( pc_new_feature_corners ) );
        m_mul_lidar_management.m_pt_cell_map_planes[ matching_pointcloud_id ].append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>( pc_new_feature_surface ) );
    }

    *( m_logger_common.get_ostream() ) << "New added regtime " << point_cloud_current_timestamp << endl;

    memcpy( m_lidar_agent[ lidar_id ].m_buffer_RT_last_incre, pc_reg.m_para_buffer_incremental, 7 * sizeof( double ) );
    m_mul_lidar_management.m_mutex_lidar_mapping[ matching_pointcloud_id ]->unlock();
    
    // ANCHOR Enable this for more thread to update the map
    if (  ! m_para_if_force_update_buffer_for_matching )
    {
        if ( m_thread_match_buff_refresh.size() < ( size_t ) m_maximum_mapping_buff_thread )
        {
            std::future<void> *m_mapping_refresh_service =
                new std::future<void>( std::async( std::launch::async, &Laser_mapping::service_update_buff_for_matching, this ) );
            m_thread_match_buff_refresh.push_back( m_mapping_refresh_service );
        }
    }
    
    // ANCHOR processing keyframe
    if ( m_loop_closure_if_enable )
    {
        std::set<Points_cloud_map<float>::Mapping_cell_ptr> cell_vec;
        m_pt_cell_map_full.append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>( current_laser_cloud_full.makeShared() ), &cell_vec );
        std::unique_lock<std::mutex> lock( m_mutex_keyframe );
        for ( auto it = m_keyframe_of_updating_list.begin(); it != m_keyframe_of_updating_list.end(); it++ )
        {
            ( *it )->add_cells( cell_vec );
        }
        
        // cout << m_keyframe_of_updating_list.front()->m_accumulate_frames << " | " << m_para_scans_of_each_keyframe << " | " << m_para_scans_between_two_keyframe << endl;
        if ( m_keyframe_of_updating_list.front()->m_accumulate_frames >= ( size_t ) m_para_scans_of_each_keyframe )
        {
            m_keyframe_of_updating_list.front()->m_ending_frame_idx = m_current_frame_index;

            m_keyframe_of_updating_list.front()->m_pose_q = m_lidar_agent[ lidar_id ].m_q_w_curr;
            m_keyframe_of_updating_list.front()->m_pose_t = m_lidar_agent[ lidar_id ].m_t_w_curr;

            m_keyframe_need_processing_list.push_back( m_keyframe_of_updating_list.front() );
            m_keyframe_of_updating_list.pop_front();
        }

        if ( m_keyframe_of_updating_list.back()->m_accumulate_frames >= ( size_t ) m_para_scans_between_two_keyframe )
        {
            m_mutex_dump_full_history.lock();

            // Full data
            for ( auto it = m_laser_cloud_full_history.begin(); it != m_laser_cloud_full_history.end(); it++ )
            {
                m_keyframe_of_updating_list.back()->m_accumulated_point_cloud_full += ( *it );
            }

            // Corners
            for ( auto it = m_laser_cloud_corner_for_keyframe.begin(); it != m_laser_cloud_corner_for_keyframe.end(); it++ )
            {
                m_keyframe_of_updating_list.back()->m_accumulated_point_cloud_corners += ( *it );
            }

            // Surface
            for ( auto it = m_laser_cloud_surface_for_keyframe.begin(); it != m_laser_cloud_surface_for_keyframe.end(); it++ )
            {
                m_keyframe_of_updating_list.back()->m_accumulated_point_cloud_surface += ( *it );
            }

            if ( m_keyframe_need_processing_list.size() > ( size_t ) m_loop_closure_maximum_keyframe_in_wating_list )
            {
                m_keyframe_need_processing_list.pop_front();
            }

            m_laser_cloud_full_history.clear();
            m_laser_cloud_corner_for_keyframe.clear();
            m_laser_cloud_surface_for_keyframe.clear();

            m_mutex_dump_full_history.unlock();
            m_keyframe_of_updating_list.push_back( std::make_shared<Maps_keyframe<float>>() );
            cout << "Number of keyframes in update lists: " << m_keyframe_of_updating_list.size() << std::endl;
        }
    }
    else
    {
        //m_pt_cell_map_full.append_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>( current_laser_cloud_full.makeShared() ) );
    }
    memcpy( pc_reg.m_para_buffer_incremental, m_lidar_agent[ lidar_id ].m_buffer_RT_last_incre, 7 * sizeof( double ) );
    m_mutex_ros_pub.lock();

    pc_reg.pointcloudAssociateToMap( current_laser_cloud_full_intensity, current_laser_cloud_full_intensity, g_if_undistore );

    // time_odom = abs(time_odom);
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg( current_laser_cloud_full_intensity, laserCloudFullRes3 );
    laserCloudFullRes3.header.frame_id = "/camera_init";
    laserCloudFullRes3.header.stamp = ros::Time().fromSec( time_odom );    
    // ANCHOR publish point cloud
    if(matching_pointcloud_id ==0 
      //&&lidar_id == 0
      )
    {
        m_pub_laser_cloud_full_res.publish( laserCloudFullRes3 ); //single_frame_with_pose_tranfromed
        m_pub_full_point_cloud_vector[lidar_id].publish( laserCloudFullRes3 );
    }
    if ( PUB_DEBUG_INFO )
    {
        pcl::PointCloud<PointType> pc_feature_pub_corners, pc_feature_pub_surface;
        sensor_msgs::PointCloud2   laserCloudMsg;

        pc_reg.pointcloudAssociateToMap( current_laser_cloud_surf_last, pc_feature_pub_surface, g_if_undistore );
        pcl::toROSMsg( pc_feature_pub_surface, laserCloudMsg );
        laserCloudMsg.header.stamp = ros::Time().fromSec( time_odom );
        laserCloudMsg.header.frame_id = "/camera_init";
        m_pub_last_surface_pts.publish( laserCloudMsg );
        pc_reg.pointcloudAssociateToMap( current_laser_cloud_corner_last, pc_feature_pub_corners, g_if_undistore );
        pcl::toROSMsg( pc_feature_pub_corners, laserCloudMsg );
        laserCloudMsg.header.stamp = ros::Time().fromSec( time_odom );
        laserCloudMsg.header.frame_id = "/camera_init";
        m_pub_last_corner_pts.publish( laserCloudMsg );
    }

    //ANCHOR Pub surface point
    if ( matching_pointcloud_id == 0 )
    {
        if ( IF_PUBLISH_SURFACE_AND_CORNER_PTS )
        {
            sensor_msgs::PointCloud2 laserCloudMsg;
            pcl::toROSMsg( *laser_cloud_surf_from_map, laserCloudMsg );
            laserCloudMsg.header.stamp = ros::Time().fromSec( time_odom );
            laserCloudMsg.header.frame_id = "/camera_init";
            m_pub_match_surface_pts.publish( laserCloudMsg );

            pcl::toROSMsg( *laser_cloud_corner_from_map, laserCloudMsg );
            laserCloudMsg.header.stamp = ros::Time().fromSec( time_odom );
            laserCloudMsg.header.frame_id = "/camera_init";
            m_pub_match_corner_pts.publish( laserCloudMsg );
        }
    }

    if ( m_if_save_to_pcd_files )
    {
        m_pcl_tools_aftmap.save_to_pcd_files( "aft_mapp", current_laser_cloud_full, m_current_frame_index );
        *( m_logger_pcd.get_ostream() ) << "Save to: " << m_pcl_tools_aftmap.m_save_file_name << endl;
    }

    //printf_line;
    //publish_current_odometry();

    static nav_msgs::Odometry odom_aft_mapped;
    static nav_msgs::Odometry odom_aft_filtered;
    odom_aft_mapped.header.frame_id = "/camera_init";
    odom_aft_mapped.header.stamp = ros::Time().fromSec( time_odom );
    odom_aft_mapped.child_frame_id = "/aft_mapped";
    Common_tools::eigen_RT_to_ros_pose( m_lidar_agent[ lidar_id ].m_q_w_curr, m_lidar_agent[ lidar_id ].m_t_w_curr, odom_aft_mapped.pose.pose );

    if ( m_lidar_agent[ lidar_id ].m_trajectory_R.size() )
    {
        odom_aft_filtered.header = odom_aft_mapped.header;
        // Common_tools::eigen_RT_to_ros_pose( m_lidar_agent[ lidar_id ].m_trajectory_R_filtered.back(), m_lidar_agent[ lidar_id ].m_trajectory_T_filtered.back(), odom_aft_filtered.pose.pose );
        m_global_ekf.get_prediction_of_idx_lidar( 0 , q_pre, t_pre);
        Common_tools::eigen_RT_to_ros_pose( q_pre, t_pre, odom_aft_filtered.pose.pose );
        m_global_ekf.get_prediction_of_idx_lidar( ekf_lidar_idx , q_pre, t_pre);
        Common_tools::eigen_RT_to_ros_pose( q_pre, t_pre, odom_aft_mapped.pose.pose );

    }
    else
    {
        odom_aft_filtered = odom_aft_mapped;
    }
    m_pub_odom_aft_mapped.publish( odom_aft_mapped );     // name: Odometry aft_mapped_to_init
    m_pub_odom_aft_filtered.publish( odom_aft_filtered ); // name: Odometry aft_mapped_to_init

    geometry_msgs::PoseStamped pose_aft_mapped, pose_aft_filtered;
    pose_aft_mapped.header = odom_aft_mapped.header;
    pose_aft_mapped.pose = odom_aft_mapped.pose.pose;

    pose_aft_filtered.header = odom_aft_mapped.header;
    pose_aft_filtered.pose = odom_aft_filtered.pose.pose;

    m_laser_path_after_mapped_vec[ lidar_id ].header.stamp = odom_aft_mapped.header.stamp;
    m_laser_path_after_mapped_vec[ lidar_id ].header.frame_id = "/camera_init";
    m_laser_path_filtered_vec[ lidar_id ].header = m_laser_path_after_mapped_vec[ lidar_id ].header;

    if ( m_current_frame_index % m_para_pub_path_downsample_factor == 0 )
    {
        m_global_ekf.get_prediction_of_idx_lidar( 0 , q_pre, t_pre);
        Common_tools::eigen_RT_to_ros_pose( q_pre, t_pre, odom_aft_filtered.pose.pose );
        
        m_laser_path_after_mapped_vec[ lidar_id ].poses.push_back( pose_aft_mapped );
        // m_laser_path_after_mapped_vec[ lidar_id ].poses.push_back( pose_aft_filtered );
        m_laser_path_filtered_vec[ lidar_id ].poses.push_back( pose_aft_filtered );
        m_pub_path_laser_aft_mapped_vec[ lidar_id ].publish( m_laser_path_after_mapped_vec[ lidar_id ] );
        m_pub_path_laser_filtered_vec[ lidar_id ].publish( m_laser_path_filtered_vec[ lidar_id ] );
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    if ( 1 )
    {
        transform.setOrigin( tf::Vector3( t_pre( 0 ),
                                          t_pre( 1 ),
                                          t_pre( 2 ) ) );
        q.setW(q_pre.w() );
        q.setX( q_pre.x() );
        q.setY( q_pre.y() );
        q.setZ( q_pre.z() );
    }
    else
    {
        transform.setOrigin( tf::Vector3( m_lidar_agent[ lidar_id ].m_t_w_curr( 0 ), m_lidar_agent[ lidar_id ].m_t_w_curr( 1 ), m_lidar_agent[ lidar_id ].m_t_w_curr( 2 ) ) );
        q.setW( m_lidar_agent[ lidar_id ].m_q_w_curr.w() );
        q.setX( m_lidar_agent[ lidar_id ].m_q_w_curr.x() );
        q.setY( m_lidar_agent[ lidar_id ].m_q_w_curr.y() );
        q.setZ( m_lidar_agent[ lidar_id ].m_q_w_curr.z() );
    }
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odom_aft_filtered.header.stamp, "/camera_init", "/aft_mapped" ) );

    m_mutex_ros_pub.unlock();
   
    *( m_logger_timer.get_ostream() ) << m_timer.toc_string( "Add new frame" ) << std::endl;
    *( m_logger_timer.get_ostream() ) << m_timer.toc_string( "Frame process" ) << std::endl;
    //printf_line;


    return 1;
};