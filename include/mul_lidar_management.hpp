#include "common.h"
#include "common_tools.h"
#include "cell_map_keyframe.hpp"
#include "custom_point_cloud_interface.hpp"
#include "scene_alignment.hpp"

#define rigid_connect_finish_time  (1.5*1.0/60.0)  // for livox lidar: 1/20/3 = 16.7ms 

// hard rigid connect
// soft connect
// frees

enum Link_type
{
    e_rigid = 0,
    e_soft = 1,
    e_free = 2
};

 // A group of lidar share the same mappings, but might be have different link type to another lidars.
 // A group of lidar own a centers, if one of the lidars in group is degenerate, the motion on this lidar on the degenerate direction is set as the same as the group center.
struct Lidar_group
{
    Link_type m_link_type;
    std::vector<int> m_lidar_member;
    size_t                          m_reserve_scan = 0;
    std::shared_ptr<Data_pair>   m_reserve_data_pair = nullptr;
    std::list<std::shared_ptr<Data_pair>>   m_data_pair_ready_for_registration_vec ;
    int  m_accumulate_frames = 0;
    void append_scans(std::shared_ptr<Data_pair>  data_pair)
    {
        if(m_reserve_data_pair == nullptr)
        {
            m_reserve_data_pair = data_pair;
            m_reserve_scan = 1;
            return;
        }
        m_reserve_data_pair->m_pc_full += data_pair->m_pc_full;
        m_reserve_data_pair->m_pc_corner += data_pair->m_pc_corner;
        m_reserve_data_pair->m_pc_plane += data_pair->m_pc_plane;
        m_reserve_scan ++;
    }

    std::shared_ptr<Data_pair> get_data_pair_for_reg()
    {
        m_accumulate_frames++;
        std::shared_ptr<Data_pair> data_ptr = m_data_pair_ready_for_registration_vec.front();
        m_data_pair_ready_for_registration_vec.pop_front();
        // cout << "Size of pointcloud :" << data_ptr->m_pc_full.m_point_size << " | "  << data_ptr->m_pc_corner.m_point_size << " | " << data_ptr->m_pc_plane.m_point_size << endl;
        return data_ptr;
    }

    int registration_lidar_scans( std::shared_ptr<Data_pair> data_pair )
    {
        if(data_pair->m_pc_full.m_point_size==0)
        {
            return 0;
        }

        // return 0, data_pair is ready for registration.
        double in_lidar_scan = data_pair->m_pc_full.m_timestamp[ 0 ];
        double reserve_lidar_time = in_lidar_scan;
        if ( m_reserve_data_pair != nullptr )
        {
            reserve_lidar_time = m_reserve_data_pair->m_pc_full.m_timestamp[ 0 ];
        }
        
        if ( m_link_type == e_rigid )
        {
            if ( abs( in_lidar_scan - reserve_lidar_time ) < rigid_connect_finish_time )
            {
                append_scans( data_pair );
                //if ( m_reserve_scan == m_lidar_member.size() )
                if(0)
                {
                    // cout << "Reach maximum bounded lidar,   size =  " << m_reserve_scan << endl;
                    m_data_pair_ready_for_registration_vec.push_back( m_reserve_data_pair );
                    m_reserve_data_pair = nullptr;
                    m_reserve_scan = 0;
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                // cout << "Out of time, ready for registation , size = " << m_reserve_scan << endl;
                m_data_pair_ready_for_registration_vec.push_back( m_reserve_data_pair );
                m_reserve_data_pair = data_pair;
                m_reserve_scan = 1;
                return 1;
            }
        }
        else
        {
            m_reserve_data_pair = data_pair;
            m_data_pair_ready_for_registration_vec.push_back( m_reserve_data_pair );
            m_reserve_scan = 1;
            return 1;
        }
    }
};

class Mul_lidar_management
{
  public:
    std::map<int, int> m_lidar_id_group; // id -> pc index
    std::vector<std::list<pcl::PointCloud<PointType>>> m_laser_cloud_corner_history_vec;
    std::vector<std::list<pcl::PointCloud<PointType>>> m_laser_cloud_surface_history_vec;
    std::vector<std::list<pcl::PointCloud<PointType>>> m_laser_cloud_full_history_vec;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> m_lidar_extrinsic_R_vec;
    std::vector<vec_3, Eigen::aligned_allocator<vec_3>>                           m_lidar_extrinsic_T_vec;
    std::vector<Lidar_group> m_lidar_group_vec;

    
    std::vector<pcl::PointCloud<PointType>> m_laser_cloud_corner_from_map_vec;
    std::vector<pcl::PointCloud<PointType>> m_laser_cloud_surf_from_map_vec;

    std::vector<pcl::PointCloud<PointType>> m_laser_cloud_corner_from_map_last_vec;
    std::vector<pcl::PointCloud<PointType>> m_laser_cloud_surf_from_map_last_vec;

    std::vector<pcl::KdTreeFLANN<PointType>> m_kdtree_corner_from_map_last;
    std::vector<pcl::KdTreeFLANN<PointType>> m_kdtree_surf_from_map_last;

    std::vector<std::shared_ptr<std::mutex>> m_mutex_buff_for_matching_corner;
    std::vector<std::shared_ptr<std::mutex>> m_mutex_buff_for_matching_surface;

    std::vector<Points_cloud_map<float>>        m_pt_cell_map_corners;
    std::vector<Points_cloud_map<float>>        m_pt_cell_map_planes;
    std::vector<int>                            m_if_have_merge;
    std::vector<std::shared_ptr<std::mutex>>    m_mutex_lidar_mapping;
    std::vector<int>                            m_lidar_frame_count;
    std::vector<int>                            m_map_index;
    Scene_alignment<float>                      m_scene_align_vec[ MAXIMUM_LIDAR_SIZE ];
    ~Mul_lidar_management(){};
    Mul_lidar_management()
    {
        m_lidar_extrinsic_R_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_lidar_extrinsic_T_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_corner_history_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_surface_history_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_full_history_vec.resize(MAXIMUM_LIDAR_SIZE);

        m_pt_cell_map_corners.resize(MAXIMUM_LIDAR_SIZE);
        m_pt_cell_map_planes.resize(MAXIMUM_LIDAR_SIZE);
        
        m_laser_cloud_corner_from_map_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_surf_from_map_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_corner_from_map_last_vec.resize(MAXIMUM_LIDAR_SIZE);
        m_laser_cloud_surf_from_map_last_vec.resize(MAXIMUM_LIDAR_SIZE);

        m_kdtree_corner_from_map_last.resize(MAXIMUM_LIDAR_SIZE);
        m_kdtree_surf_from_map_last.resize( MAXIMUM_LIDAR_SIZE );
        m_lidar_frame_count.resize( MAXIMUM_LIDAR_SIZE );
        m_if_have_merge.resize( MAXIMUM_LIDAR_SIZE );
        m_map_index.resize(MAXIMUM_LIDAR_SIZE);
        for ( size_t i = 0; i < MAXIMUM_LIDAR_SIZE; i++ )
        {
            m_lidar_frame_count[ i ] = 0;
            m_lidar_extrinsic_R_vec[ i ] = Eigen::Quaterniond::Identity();
            m_lidar_extrinsic_T_vec[ i ].setZero();

            m_mutex_buff_for_matching_corner.push_back( std::make_shared<std::mutex>() );
            m_mutex_buff_for_matching_surface.push_back( std::make_shared<std::mutex>() );
            m_if_have_merge[ i ] = 0;
            m_mutex_lidar_mapping.push_back( std::make_shared<std::mutex>() );
            std::string temp_log_dir = std::string( TEMP_LOG_SAVE_DIR ).append( "/temp_lidar_" ).append( std::to_string( i ) );

            Common_tools::create_dir( temp_log_dir );
            m_scene_align_vec[i].init( temp_log_dir );
        }

        

        init();
    };
    
    void set_extrinsic(std::vector<double> & intrinsic_data_buffer) 
    {
        // The first three LiDAR are of three unit of livox-Mid 100.
        for ( size_t i = 0; i < 3; i++ )
        {
            m_lidar_extrinsic_R_vec[ i ].coeffs() << intrinsic_data_buffer[ 0 ], intrinsic_data_buffer[ 1 ], intrinsic_data_buffer[ 2 ], intrinsic_data_buffer[ 3 ];
            m_lidar_extrinsic_T_vec[ i ] << intrinsic_data_buffer[ 4 ], intrinsic_data_buffer[ 5 ], intrinsic_data_buffer[ 6 ];
        }
        for(size_t i = 1 ; i < 5; i++)
        {
            m_lidar_extrinsic_R_vec[ 2 + i ].coeffs() << intrinsic_data_buffer[ i * 7 + 0 ], intrinsic_data_buffer[ i * 7 + 1 ], intrinsic_data_buffer[ i * 7 + 2 ], intrinsic_data_buffer[ i * 7 + 3 ];
            m_lidar_extrinsic_T_vec[ 2 + i ] << intrinsic_data_buffer[ i * 7 + 4 ], intrinsic_data_buffer[ i * 7 + 5 ], intrinsic_data_buffer[ i * 7 + 6 ];
        }

        for ( size_t i = 0; i < MAXIMUM_LIDAR_SIZE; i++ )
        { 
            cout << i << "-th lidar extrinsic Q[ " << m_lidar_extrinsic_R_vec[ i ].coeffs().transpose() << " ], "
                 << "T [ " << m_lidar_extrinsic_T_vec[ i ].transpose() << " ]" << endl;
        }
    }

    void init()
    {
        if ( 1 )
        {
            Lidar_group lidar_group;
            // lidar 0, 1, 2 is set as rigid connect
            lidar_group.m_link_type = e_rigid;
            lidar_group.m_lidar_member.push_back( 0 );
            lidar_group.m_lidar_member.push_back( 1 );
            lidar_group.m_lidar_member.push_back( 2 );
            m_lidar_group_vec.push_back( lidar_group );
            m_lidar_id_group.insert( std::make_pair( 0, 0 ) );
            m_lidar_id_group.insert( std::make_pair( 1, 0 ) );
            m_lidar_id_group.insert( std::make_pair( 2, 0 ) );

        }

        // lidar 3, 4, 5, 6 is set as soft connect
        for ( size_t idx = 0; idx < 7; idx++ )
        {
            Lidar_group lidar_group;
            lidar_group.m_link_type = e_soft;
            lidar_group.m_lidar_member.push_back( 3 + idx );
            m_lidar_group_vec.push_back( lidar_group );
            m_lidar_id_group.insert( std::make_pair( 3 + idx, 1 + idx ) );
            
        }
        m_map_index[ 0 ] = 0;
        m_map_index[ 1 ] = 0;
        m_map_index[ 2 ] = 0;
        m_map_index[ 3 ] = 3;
        m_map_index[ 4 ] = 4;
        m_map_index[ 5 ] = 5;
        m_map_index[ 6 ] = 6;

        cout << "===== Finish initialization =====" << endl;
    }

    int find_group_idx( int lidar_idx)
    {
        return m_lidar_id_group.find(lidar_idx)->second;
    };

    void get_extrinsic(int lidar_idx, Eigen::Quaterniond & R_mat_e, Eigen::Matrix<double, 3, 1> t_e )
    {
        int idx = find_group_idx(lidar_idx);
        R_mat_e = m_lidar_extrinsic_R_vec[idx];
        t_e = m_lidar_extrinsic_T_vec[idx]; 
    };

    void load_extrinsic_from_file(string file_name)
    {

    }

    int registration_lidar_scans( std::shared_ptr<Data_pair >  data_pair)
    {
        // return 0, data_pair is ready for registration.
        int group_idx= find_group_idx(data_pair->m_pc_full.m_lidar_id);
        if ( 0 )
        {
            cout << "Accumulate frame count: ";
            for ( size_t idx = 0; idx < 7; idx++ )
            {
                cout << m_lidar_group_vec[ idx ].m_accumulate_frames << " | ";
            }
            cout << endl;
        }
        return m_lidar_group_vec[group_idx].registration_lidar_scans(data_pair);
    }

    std::shared_ptr<Data_pair > get_data_pair_for_registraction(std::shared_ptr<Data_pair >  data_pair)
    {
        int group_idx= find_group_idx(data_pair->m_pc_full.m_lidar_id);
        std::shared_ptr<Data_pair > res_prt = m_lidar_group_vec[group_idx].get_data_pair_for_reg();
        if ( m_lidar_group_vec[ group_idx ].m_link_type == e_rigid )
        {
            if ( 1 )
            {
                res_prt->m_pc_full.m_lidar_id = group_idx;
                res_prt->m_pc_corner.m_lidar_id = group_idx;
                res_prt->m_pc_plane.m_lidar_id = group_idx;
            }
            else
            {
                res_prt->m_pc_full.m_lidar_id = 0;
                res_prt->m_pc_corner.m_lidar_id = 0;
                res_prt->m_pc_plane.m_lidar_id = 0;
            }
        }
        return res_prt;
    }


    
    template <typename T>
    void find_overlap_of_two_map( Points_cloud_map< T > & map_a, Points_cloud_map< T  > & map_b, 
                                std::set< std::shared_ptr< typename Points_cloud_map< T >::Mapping_cell > > & cell_vec_overlap_a,
                                std::set< std::shared_ptr< typename Points_cloud_map< T >::Mapping_cell > > & cell_vec_overlap_b  )
    {
        std::set< std::shared_ptr< typename Points_cloud_map< T >::Mapping_cell > > cell_vec_a, cell_vec_b;
        cell_vec_overlap_a.clear();
        cell_vec_overlap_b.clear();
        assert(map_b.m_resolution = map_a.m_resolution);
        //cell_vec_a = map_a.m_cell_vec;
        for(auto cell_ptr_a: map_a.m_cell_vec)
        {
            auto cell_ptr_b  = map_b.find_cell(cell_ptr_a->get_center(), 0, 0);
            if(cell_ptr_b!=nullptr )
            {
                cell_vec_overlap_a.insert(cell_ptr_a);
                cell_vec_overlap_b.insert(cell_ptr_b);
                    }
        }
        assert(cell_vec_overlap_b.size() == cell_vec_overlap_a.size());
    }

};
