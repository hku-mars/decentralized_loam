#include "scene_alignment.hpp"

template class Scene_alignment<double>;
template class Scene_alignment<float>;

template <class PT_DATA_TYPE>
void Scene_alignment<PT_DATA_TYPE>::set_downsample_resolution( const float &line_res, const float &plane_res )
{
    m_line_res = line_res;
    m_plane_res = plane_res;
    m_down_sample_filter_line_source.setLeafSize( m_line_res, m_line_res, m_line_res );
    m_down_sample_filter_surface_source.setLeafSize( m_plane_res, m_plane_res, m_plane_res );
    m_down_sample_filter_line_target.setLeafSize( m_line_res, m_line_res, m_line_res );
    m_down_sample_filter_surface_target.setLeafSize( m_plane_res, m_plane_res, m_plane_res );
}

template <class PT_DATA_TYPE>
void Scene_alignment<PT_DATA_TYPE>::init( std::string path )
{
    m_save_path = path.append( "/scene_align" );
    Common_tools::create_dir( m_save_path );
    file_logger_common.set_log_dir( m_save_path );
    file_logger_timer.set_log_dir( m_save_path );
    file_logger_common.init( "common.log" );
    file_logger_timer.init( "timer.log" );

    m_pc_reg.ICP_LINE = 0;
    m_pc_reg.m_logger_common = &file_logger_common;
    m_pc_reg.m_logger_timer = &file_logger_timer;
    m_pc_reg.m_timer = &timer;
    m_pc_reg.m_para_cere_max_iterations = m_maximum_icp_iteration;
    m_pc_reg.m_max_final_cost = 20000;
    m_pc_reg.m_para_max_speed = 1000.0;
    m_pc_reg.m_para_max_angular_rate = 360 * 57.3;
    m_pc_reg.m_para_icp_max_iterations = 10;
    m_pc_reg.m_para_cere_max_iterations = 20;
    m_pc_reg.m_inliner_dis = 0.2;
}

template <class PT_DATA_TYPE>
void Scene_alignment<PT_DATA_TYPE>::dump_file_name( std::string                 save_file_name,
                                                    std::map<int, std::string> &map_filename )
{
    FILE *fp = fopen( save_file_name.c_str(), "w+" );
    if ( fp != NULL )
    {
        for ( auto it = map_filename.begin(); it != map_filename.end(); it++ )
        {
            fprintf( fp, "%d %s\r\n", it->first, it->second.c_str() );
        }
        fclose( fp );
    }
};

template <class PT_DATA_TYPE>
int Scene_alignment<PT_DATA_TYPE>::find_tranfrom_of_two_mappings( std::shared_ptr<Maps_keyframe<PT_DATA_TYPE>> keyframe_a,
                                                                  std::shared_ptr<Maps_keyframe<PT_DATA_TYPE>> keyframe_b,
                                                                  int                                          if_save,
                                                                  std::string                                  mapping_save_path )
{
    return find_tranfrom_of_two_mappings( keyframe_a.get(), keyframe_b.get(), if_save, mapping_save_path );
};

template <class PT_DATA_TYPE>
int Scene_alignment<PT_DATA_TYPE>::find_tranfrom_of_two_mappings( Maps_keyframe<PT_DATA_TYPE> *keyframe_a,
                                                                  Maps_keyframe<PT_DATA_TYPE> *keyframe_b,
                                                                  int                          if_save,
                                                                  std::string                  mapping_save_path , 
                                                                  int if_all_plane)
{
    pcl::PointCloud< pcl_pt > sourece_pt_line, sourece_pt_plane, target_pt_line, target_pt_plane;
    if ( if_all_plane == 0 )
    {
        sourece_pt_line = keyframe_a->extract_specify_points( Feature_type::e_feature_line );
        sourece_pt_plane = keyframe_a->extract_specify_points( Feature_type::e_feature_plane );

        target_pt_line = keyframe_b->extract_specify_points( Feature_type::e_feature_line );
        target_pt_plane = keyframe_b->extract_specify_points( Feature_type::e_feature_plane );
    }
    else
    {
        sourece_pt_plane = keyframe_a->get_all_pointcloud();
        target_pt_plane = keyframe_b->get_all_pointcloud();
        //sourece_pt_line = sourece_pt_plane;
        //target_pt_line = target_pt_plane;
    }
    pcl::PointCloud< pcl_pt > all_pt_a = keyframe_a->get_all_pointcloud();
    pcl::PointCloud< pcl_pt > all_pt_b = keyframe_b->get_all_pointcloud();

    pcl::PointCloud< pcl_pt > sourece_pt_line_ds, sourece_pt_plane_ds; // Point cloud of downsampled
    pcl::PointCloud< pcl_pt > target_pt_line_ds, target_pt_plane_ds;

    m_down_sample_filter_line_source.setInputCloud( sourece_pt_line.makeShared() );
    m_down_sample_filter_surface_source.setInputCloud( sourece_pt_plane.makeShared() );
    m_down_sample_filter_line_target.setInputCloud( target_pt_line.makeShared() );
    m_down_sample_filter_surface_target.setInputCloud( target_pt_plane.makeShared() );

    m_pc_reg.m_current_frame_index = 10000000;
    m_pc_reg.m_q_w_curr.setIdentity();
    m_pc_reg.m_q_w_last.setIdentity();
    m_pc_reg.m_t_w_last.setZero();
    m_pc_reg.m_para_icp_max_iterations = m_maximum_icp_iteration;
    m_pc_reg.m_para_cere_max_iterations = 50;
    m_pc_reg.m_para_cere_prerun_times = 2;
    m_pc_reg.m_if_degenerate = 0;
    m_pc_reg.m_inlier_threshold =  1e3;
    m_pc_reg.m_inlier_ratio = 0.80;
    m_pc_reg.m_maximum_allow_residual_block = m_para_scene_alignments_maximum_residual_block;

    m_pc_reg.m_if_verbose_screen_printf = m_if_verbose_screen_printf;

    Eigen::Matrix< double, 3, 1 > transform_T = ( keyframe_a->get_center() - keyframe_b->get_center() ).template cast< double >();
    Eigen::Quaterniond transform_R = Eigen::Quaterniond::Identity();
    m_pc_reg.m_t_w_incre = transform_T;
    m_pc_reg.m_t_w_curr = transform_T;
    Common_tools::Timer timer;
    timer.tic( "Total" );
    for ( int scale = 8; scale >= 0; scale -= 4 )
    {

        timer.tic( "Each omp" );

        float line_res = m_line_res * scale;
        float plane_res = m_plane_res * scale;
        if ( line_res < m_line_res )
        {
            line_res = m_line_res;
        }

        if ( plane_res < m_plane_res )
        {
            plane_res = m_plane_res;
            m_pc_reg.m_para_icp_max_iterations = m_maximum_icp_iteration * 2;
        }
        m_down_sample_filter_line_source.setLeafSize( line_res, line_res, line_res );
        m_down_sample_filter_surface_source.setLeafSize( plane_res, plane_res, plane_res );
        m_down_sample_filter_line_target.setLeafSize( line_res, line_res, line_res );
        m_down_sample_filter_surface_target.setLeafSize( plane_res, plane_res, plane_res );

        m_down_sample_filter_line_source.filter( sourece_pt_line_ds );
        m_down_sample_filter_surface_source.filter( sourece_pt_plane_ds );

        m_down_sample_filter_line_target.filter( target_pt_line_ds );
        m_down_sample_filter_surface_target.filter( target_pt_plane_ds );

        screen_out << "Source pt line size = " << sourece_pt_line_ds.size() << " , plane size = " << sourece_pt_plane_ds.size() << std::endl;
        screen_out << "Target pt line size = " << target_pt_line_ds.size() << " , plane size = " << target_pt_plane_ds.size() << std::endl;

        m_pc_reg.find_out_incremental_transfrom( sourece_pt_line_ds.makeShared(), sourece_pt_plane_ds.makeShared(), target_pt_line_ds.makeShared(), target_pt_plane_ds.makeShared() );
        screen_out << "===*** Result of pc_reg: ===*** " << endl;
        screen_out << "Resoulation = " << line_res << " -- " << plane_res << endl;
        screen_out << "Q_incre is: " << m_pc_reg.m_q_w_incre.coeffs().transpose() << endl;
        screen_out << "T_incre is: " << m_pc_reg.m_t_w_incre.transpose() << endl;
        screen_out << "Q_curr is: " << m_pc_reg.m_q_w_curr.coeffs().transpose() << endl;
        screen_out << "T_curr is: " << m_pc_reg.m_t_w_curr.transpose() << endl;
        screen_out << timer.toc_string( "Each omp" ) << std::endl;

        if ( m_pc_reg.m_inlier_threshold > m_accepted_threshold * 2 )
            break;
    }
    screen_out << timer.toc_string( "Total" ) << std::endl;

    if ( if_save )
    {
        Points_cloud_map<PT_DATA_TYPE> temp_res;
        double cell_resolution = keyframe_a->m_map_pt_cell.begin()->second->m_resolution *  2;
        temp_res.set_resolution(cell_resolution);
        auto all_pt_temp = PCL_TOOLS::pointcloud_transfrom<double, pcl_pt>( all_pt_b, m_pc_reg.m_q_w_curr.toRotationMatrix(), m_pc_reg.m_t_w_curr );
        temp_res.set_point_cloud( PCL_TOOLS::pcl_pts_to_eigen_pts<PT_DATA_TYPE, pcl_pt>( all_pt_temp.makeShared() ) );

        std::string save_path;
        if ( mapping_save_path.compare( std::string( " " ) ) == 0 )
        {
            save_path = m_save_path;
        }
        else
        {
            save_path = mapping_save_path;
        }

        keyframe_a->save_to_file( save_path, std::to_string( pair_idx ).append( "_a.json" ) );
        keyframe_b->save_to_file( save_path, std::to_string( pair_idx ).append( "_b.json" ) );
        temp_res.save_to_file( save_path, std::to_string( pair_idx ).append( "_c.json" ) );

        temp_res.clear_data();
        pair_idx++;
    }

    sourece_pt_line.clear();
    sourece_pt_plane.clear();
    all_pt_a.clear();
    all_pt_b.clear();
    target_pt_line_ds.clear();
    sourece_pt_plane_ds.clear();
    target_pt_line_ds.clear();
    target_pt_plane_ds.clear();

    return m_pc_reg.m_inlier_threshold;
};
