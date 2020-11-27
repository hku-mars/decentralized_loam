#include "cell_map_keyframe.hpp"

template class Points_cloud_map<float>;
template class Points_cloud_map<double>;

template <class DATA_TYPE>
Points_cloud_map<DATA_TYPE>::Points_cloud_map()
{
    m_mapping_mutex = std::make_shared<std::mutex>();
    m_mutex_addcell = std::make_shared<std::mutex>();
    m_octotree_mutex = std::make_shared<std::mutex>();
    m_x_min = std::numeric_limits<DATA_TYPE>::max();
    m_y_min = std::numeric_limits<DATA_TYPE>::max();
    m_z_min = std::numeric_limits<DATA_TYPE>::max();

    m_x_max = std::numeric_limits<DATA_TYPE>::min();
    m_y_max = std::numeric_limits<DATA_TYPE>::min();
    m_z_max = std::numeric_limits<DATA_TYPE>::min();
    m_pcl_cells_center->reserve( 1e5 );
    set_resolution( 1.0 );
    m_if_verbose_screen_printf = 1;
    m_current_frame_idx = 0;
};

template <class DATA_TYPE>
Points_cloud_map<DATA_TYPE>::~Points_cloud_map()
{
    m_mapping_mutex->try_lock();
    m_mapping_mutex->unlock();

    m_mutex_addcell->try_lock();
    m_mutex_addcell->unlock();

    m_octotree_mutex->try_lock();
    m_octotree_mutex->unlock();
    m_octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>( 0.0001 );
}

template <class DATA_TYPE>
FUNC_T void Points_cloud_map<DATA_TYPE>::set_update_mean_and_cov_incrementally( int flag  )
{
    m_if_incremental_update_mean_and_cov = flag;
}

template <class DATA_TYPE>
FUNC_T int Points_cloud_map<DATA_TYPE>::get_cells_size()
{
    return m_map_pt_cell.size();
}

template <class DATA_TYPE>
FUNC_T typename Points_cloud_map<DATA_TYPE>::PT_TYPE Points_cloud_map<DATA_TYPE>::find_cell_center( const PT_TYPE &pt )
{
    PT_TYPE   cell_center;
    DATA_TYPE box_size = m_resolution * 1.0;
    DATA_TYPE half_of_box_size = m_resolution * 0.5;

    // cell_center( 0 ) = ( std::round( ( pt( 0 ) - m_x_min - half_of_box_size ) / box_size ) ) * box_size + m_x_min + half_of_box_size;
    // cell_center( 1 ) = ( std::round( ( pt( 1 ) - m_y_min - half_of_box_size ) / box_size ) ) * box_size + m_y_min + half_of_box_size;
    // cell_center( 2 ) = ( std::round( ( pt( 2 ) - m_z_min - half_of_box_size ) / box_size ) ) * box_size + m_z_min + half_of_box_size;

    cell_center( 0 ) = ( std::round( ( pt( 0 ) - half_of_box_size ) / box_size ) ) * box_size + half_of_box_size;
    cell_center( 1 ) = ( std::round( ( pt( 1 ) - half_of_box_size ) / box_size ) ) * box_size + half_of_box_size;
    cell_center( 2 ) = ( std::round( ( pt( 2 ) - half_of_box_size ) / box_size ) ) * box_size + half_of_box_size;

    return cell_center;
}

template <class DATA_TYPE>
FUNC_T void Points_cloud_map<DATA_TYPE>::clear_data()
{
    for ( Map_pt_cell_it it = m_map_pt_cell.begin(); it != m_map_pt_cell.end(); it++ )
    {
        it->second->clear_data();
    }
    m_map_pt_cell.clear();
    m_cell_vec.clear();
    m_pcl_cells_center->clear();
    // m_octree.deleteTree(); // will be crash
}

template <class DATA_TYPE>
FUNC_T void Points_cloud_map<DATA_TYPE>::set_point_cloud( const std::vector<PT_TYPE> &input_pt_vec, std::set<std::shared_ptr<Mapping_cell>> *cell_vec, int if_octree )
{
    clear_data();
    for ( size_t i = 0; i < input_pt_vec.size(); i++ )
    {
        m_x_min = std::min( input_pt_vec[ i ]( 0 ), m_x_min );
        m_y_min = std::min( input_pt_vec[ i ]( 1 ), m_y_min );
        m_z_min = std::min( input_pt_vec[ i ]( 2 ), m_z_min );

        m_x_max = std::max( input_pt_vec[ i ]( 0 ), m_x_max );
        m_y_max = std::max( input_pt_vec[ i ]( 1 ), m_y_max );
        m_z_max = std::max( input_pt_vec[ i ]( 2 ), m_z_max );
    }
    if ( cell_vec != nullptr )
    {
        cell_vec->clear();
    }
    for ( size_t i = 0; i < input_pt_vec.size(); i++ )
    {
        std::shared_ptr<Mapping_cell> cell = find_cell( input_pt_vec[ i ], 1, 1 );
        cell->append_pt( input_pt_vec[ i ] );
        if ( cell_vec != nullptr )
        {
            cell_vec->insert( cell );
        }
    }

    if ( if_octree )
    {
        m_octree.setInputCloud( m_pcl_cells_center );
        m_octree.addPointsFromInputCloud();
    }
    screen_out << "*** set_point_cloud octree initialization finish ***" << std::endl;
    m_initialized = true;
    m_current_frame_idx++;
}

template <class DATA_TYPE>
FUNC_T void Points_cloud_map<DATA_TYPE>::append_cloud( const std::vector<PT_TYPE> &input_pt_vec, std::set<Mapping_cell_ptr> *cell_vec )
{
    // ENABLE_SCREEN_PRINTF;
    std::map<Mapping_cell_ptr, int> appeared_cell_count;
    m_timer.tic( __FUNCTION__ );
    m_mapping_mutex->lock();
    int current_size = get_cells_size();
    if ( current_size == 0 )
    {
        set_point_cloud( input_pt_vec, cell_vec );
        m_mapping_mutex->unlock();
    }
    else
    {
        m_mapping_mutex->unlock();
        if ( cell_vec != nullptr )
        {
            cell_vec->clear();
        }
        for ( size_t i = 0; i < input_pt_vec.size(); i++ )
        {
            Mapping_cell_ptr cell = find_cell( input_pt_vec[ i ], 1, 1 );
            cell->append_pt( input_pt_vec[ i ] );
            if ( cell_vec != nullptr )
            {
                auto it = appeared_cell_count.find( cell );
                if ( it != appeared_cell_count.end() )
                {
                    it->second++;
                }
                else
                {
                    appeared_cell_count.insert( std::make_pair( cell, 1 ) );
                }
            }
        }

        if ( cell_vec != nullptr )
        {
            for ( auto it = appeared_cell_count.begin(); it != appeared_cell_count.end(); it++ )
            {
                if ( it->second >= 3 )
                {
                    cell_vec->insert( it->first );
                }
            }
        }
    }
    m_current_frame_idx++;
    screen_out << "Input points size: " << input_pt_vec.size() << ", "
               << "add cell number: " << get_cells_size() - current_size << ", "
               << "curren cell number: " << m_map_pt_cell.size() << std::endl;
    screen_out << m_timer.toc_string( __FUNCTION__ ) << std::endl;
}

template <class DATA_TYPE>
FUNC_T DATA_TYPE Points_cloud_map<DATA_TYPE>::get_resolution()
{
    return m_resolution * 2;
}

template <class DATA_TYPE>
FUNC_T typename Points_cloud_map<DATA_TYPE>::Mapping_cell_ptr Points_cloud_map<DATA_TYPE>::add_cell( const PT_TYPE &cell_center )
{
    std::unique_lock<std::mutex> lock( *m_mutex_addcell );
    Map_pt_cell_it               it = m_map_pt_cell.find( cell_center );
    if ( it != m_map_pt_cell.end() )
    {
        return it->second;
    }

    Mapping_cell_ptr cell = std::make_shared<Mapping_cell>( cell_center, ( DATA_TYPE ) m_resolution );
    cell->m_create_frame_idx = m_current_frame_idx;
    cell->m_last_update_frame_idx = m_current_frame_idx;
    cell->m_if_incremental_update_mean_and_cov = m_if_incremental_update_mean_and_cov;
    m_map_pt_cell.insert( std::make_pair( cell_center, cell ) );

    if ( m_initialized == false )
    {
        m_pcl_cells_center->push_back( pcl::PointXYZ( cell->m_center( 0 ), cell->m_center( 1 ), cell->m_center( 2 ) ) );
    }
    else
    {
        std::unique_lock<std::mutex> lock( *m_octotree_mutex );
        m_octree.addPointToCloud( pcl::PointXYZ( cell->m_center( 0 ), cell->m_center( 1 ), cell->m_center( 2 ) ), m_pcl_cells_center );
    }

    m_cell_vec.insert( cell );
    return cell;
}

template <class DATA_TYPE>
FUNC_T typename Points_cloud_map<DATA_TYPE>::Mapping_cell_ptr Points_cloud_map<DATA_TYPE>::find_cell( const PT_TYPE &pt, int if_add , int if_treat_revisit  )
{
    PT_TYPE        cell_center = find_cell_center( pt );
    Map_pt_cell_it it = m_map_pt_cell.find( cell_center );
    if ( it == m_map_pt_cell.end() )
    {
        if ( if_add )
        {
            Mapping_cell_ptr cell_ptr = add_cell( cell_center );
            return cell_ptr;
        }
        else
        {
            return nullptr;
        }
    }
    else
    {
        if ( if_treat_revisit )
        {
            if ( m_current_frame_idx - it->second->m_last_update_frame_idx < m_minimum_revisit_threshold )
            {
                it->second->m_last_update_frame_idx = m_current_frame_idx;
                return it->second;
            }
            else
            {
                // Avoid confilcts of revisited
                // ENABLE_SCREEN_PRINTF;
                // screen_out << "!!!!! Cell revisit, curr_idx = " << m_current_frame_idx << " ,last_idx = " << it->second->m_last_update_frame_idx << std::endl;

                Mapping_cell_ptr new_cell = std::make_shared<Mapping_cell>( it->second->get_center(), ( DATA_TYPE ) m_resolution );
                m_cell_vec.insert( new_cell );
                new_cell->m_previous_visited_cell = it->second;
                it->second = new_cell;
                it->second->m_create_frame_idx = m_current_frame_idx;
                it->second->m_last_update_frame_idx = m_current_frame_idx;
                //screen_out << ", find cell addr = " << ( void * ) find_cell( pt ) << std::endl;
            }
        }
        return it->second;
    }
}

template <class DATA_TYPE>
FUNC_T std::string Points_cloud_map<DATA_TYPE>::to_json_string( int &avail_cell_size )
{
    std::string str;
    str.reserve( m_map_pt_cell.size() * 1e4 );
    std::stringstream str_s( str );
    str_s << "[";
    avail_cell_size = 0;
    for ( Map_pt_cell_it it = m_map_pt_cell.begin(); it != m_map_pt_cell.end(); )
    {
        Mapping_cell_ptr cell = it->second;

        if ( avail_cell_size != 0 )
        {
            str_s << ",";
        }
        str_s << cell->to_json_string();
        avail_cell_size++;

        it++;
        if ( it == m_map_pt_cell.end() )
        {
            break;
        }
    }
    str_s << "]";
    return str_s.str();
}

template <class DATA_TYPE>
FUNC_T void Points_cloud_map<DATA_TYPE>::save_to_file( const std::string &path , const std::string &file_name )
{
    ENABLE_SCREEN_PRINTF;
    std::stringstream str_ss;
    Common_tools::create_dir( path );
    if ( file_name.compare( "" ) == 0 )
    {
        str_ss << path << "/" << std::setprecision( 3 ) << "mapping.json";
    }
    else
    {
        str_ss << path << "/" << file_name.c_str();
    }
    std::fstream ofs;
    ofs.open( str_ss.str().c_str(), std::ios_base::out );
    screen_out << "Save to " << str_ss.str();
    if ( ofs.is_open() )
    {
        int avail_cell_size = 0;
        ofs << to_json_string( avail_cell_size );
        ofs.close();
        screen_out << " Successful. Number of cell = " << avail_cell_size << std::endl;
    }
    else
    {
        screen_out << " Fail !!!" << std::endl;
    }
}


template <class DATA_TYPE>
FUNC_T int Points_cloud_map<DATA_TYPE>::load_mapping_from_file( const std::string &file_name  )
{
    Common_tools::Timer timer;
    timer.tic( "Load mapping from json file" );
    FILE *fp = fopen( file_name.c_str(), "r" );
    if ( fp == nullptr )
    {
        std::cout << "load_mapping_from_file: " << file_name << " fail!" << std::endl;
        return 0;
    }
    else
    {
        m_json_file_name = file_name;
        char                      readBuffer[ 1 << 16 ];
        rapidjson::FileReadStream is( fp, readBuffer, sizeof( readBuffer ) );

        rapidjson::Document doc;
        doc.ParseStream( is );
        if ( doc.HasParseError() )
        {
            printf( "GetParseError, err_code =  %d\n", doc.GetParseError() );
            return 0;
        }

        DATA_TYPE *pt_vec_data;
        size_t     pt_num;
        for ( unsigned int i = 0; i < doc.Size(); ++i )
        {
            if ( i == 0 )
            {
                set_resolution( doc[ i ][ "Res" ].GetDouble() * 2.0 );
            }

            typename Points_cloud_map<DATA_TYPE>::Mapping_cell_ptr cell = add_cell( Eigen::Matrix<DATA_TYPE, 3, 1>( get_json_array<DATA_TYPE>( doc[ i ][ "Center" ].GetArray() ) ) );

            cell->m_mean = Eigen::Matrix<COMP_TYPE, 3, 1>( get_json_array<COMP_TYPE>( doc[ i ][ "Mean" ].GetArray() ) );
            cell->m_cov_mat = Eigen::Matrix<COMP_TYPE, 3, 3>( get_json_array<COMP_TYPE>( doc[ i ][ "Cov" ].GetArray() ) );
            cell->m_icov_mat = Eigen::Matrix<COMP_TYPE, 3, 3>( get_json_array<COMP_TYPE>( doc[ i ][ "Icov" ].GetArray() ) );
            cell->m_eigen_vec = Eigen::Matrix<COMP_TYPE, 3, 3>( get_json_array<COMP_TYPE>( doc[ i ][ "Eig_vec" ].GetArray() ) );
            cell->m_eigen_val = Eigen::Matrix<COMP_TYPE, 3, 1>( get_json_array<COMP_TYPE>( doc[ i ][ "Eig_val" ].GetArray() ) );

            pt_num = doc[ i ][ "Pt_num" ].GetInt();
            cell->m_points_vec.resize( pt_num );
            pt_vec_data = get_json_array<DATA_TYPE>( doc[ i ][ "Pt_vec" ].GetArray() );
            for ( size_t pt_idx = 0; pt_idx < pt_num; pt_idx++ )
            {
                cell->m_points_vec[ pt_idx ] << pt_vec_data[ pt_idx * 3 + 0 ], pt_vec_data[ pt_idx * 3 + 1 ], pt_vec_data[ pt_idx * 3 + 2 ];
                cell->m_xyz_sum = cell->m_xyz_sum + cell->m_points_vec[ pt_idx ].template cast<COMP_TYPE>();
            }
            delete pt_vec_data;
        }
        fclose( fp );

        std::cout << timer.toc_string( "Load mapping from json file" ) << std::endl;
        return m_map_pt_cell.size();
    }
}

template <class DATA_TYPE>
FUNC_T std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> Points_cloud_map<DATA_TYPE>::load_pts_from_file( const std::string &file_name  )
{
    Common_tools::Timer timer;
    timer.tic( "Load points from json file" );
    FILE *                                      fp = fopen( file_name.c_str(), "r" );
    std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> res_vec;
    if ( fp == nullptr )
    {
        return res_vec;
    }
    else
    {
        m_json_file_name = file_name;
        char                      readBuffer[ 1 << 16 ];
        rapidjson::FileReadStream is( fp, readBuffer, sizeof( readBuffer ) );

        rapidjson::Document doc;
        doc.ParseStream( is );
        if ( doc.HasParseError() )
        {
            printf( "GetParseError, error code = %d\n", doc.GetParseError() );
            return res_vec;
        }

        DATA_TYPE *pt_vec_data;
        size_t     pt_num;

        for ( unsigned int i = 0; i < doc.Size(); ++i )
        {
            std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> pt_vec_cell;
            pt_num = doc[ i ][ "Pt_num" ].GetInt();
            pt_vec_cell.resize( pt_num );
            pt_vec_data = get_json_array<DATA_TYPE>( doc[ i ][ "Pt_vec" ].GetArray() );
            for ( size_t pt_idx = 0; pt_idx < pt_num; pt_idx++ )
            {
                pt_vec_cell[ pt_idx ] << pt_vec_data[ pt_idx * 3 + 0 ], pt_vec_data[ pt_idx * 3 + 1 ], pt_vec_data[ pt_idx * 3 + 2 ];
            }
            res_vec.insert( res_vec.end(), pt_vec_cell.begin(), pt_vec_cell.end() );
        }
        fclose( fp );
        std::cout << "****** Load point from:" << file_name << "  successful ****** " << std::endl;
        std::cout << timer.toc_string( "Load points from json file" ) << std::endl;
    }
    return res_vec;
}

//template <typename T>
template <class DATA_TYPE>
FUNC_T std::vector<typename Points_cloud_map<DATA_TYPE>::PT_TYPE> Points_cloud_map<DATA_TYPE>::query_point_cloud( std::vector<Mapping_cell *> &cell_vec )
{
    std::vector<std::vector<PT_TYPE>> pt_vec_vec;
    pt_vec_vec.reserve( 1000 );
    for ( int i = 0; i < cell_vec.size(); i++ )
    {
        pt_vec_vec.push_back( cell_vec[ i ]->get_pointcloud_eigen() );
    }
    return Common_tools::vector_2d_to_1d( pt_vec_vec );
}

template <class DATA_TYPE>
FUNC_T pcl::PointCloud<pcl_pt> Points_cloud_map<DATA_TYPE>::extract_specify_points( Feature_type select_type )
{
    pcl::PointCloud<pcl_pt> res_pt;
    int                     cell_is_selected_type = 0;
    //for ( size_t i = 0; i < m_cell_vec.size(); i++ )
    for(auto cell_ptr: m_cell_vec)
    {
        if ( cell_ptr->m_feature_type == select_type )
        {
            cell_is_selected_type++;
            res_pt += cell_ptr->get_pointcloud();
        }
    }

    return res_pt;
}

template <class DATA_TYPE>
FUNC_T pcl::PointCloud<pcl_pt> Points_cloud_map<DATA_TYPE>::get_all_pointcloud()
{
    pcl::PointCloud<pcl_pt> res_pt;
    for(auto cell_ptr: m_cell_vec)
    {

        res_pt += cell_ptr->get_pointcloud();
    }
    return res_pt;
}