#include "cell_map_keyframe.hpp"

template class Maps_keyframe<float>;
template class Maps_keyframe<double>;

template <class DATA_TYPE>
Maps_keyframe<DATA_TYPE>::Maps_keyframe()
{
    m_keyframe_idx = std::make_shared<std::mutex>();
    m_accumulated_point_cloud_full.reserve( 5000 * 500 / 3 ); //each frame have 5000/3 pts, about maximum 500 frame of accumulation.
}

template <class DATA_TYPE>
Maps_keyframe<DATA_TYPE>::~Maps_keyframe()
{
}

template <class DATA_TYPE>
void Maps_keyframe<DATA_TYPE>::make_index_in_matrix_range( int &idx, int maximum_range )
{
    if ( idx < 0 )
    {
        std::cout << "Idx < 0 !!!" << std::endl;
        idx = 0;
    }
    if ( idx >= maximum_range )
    {
        std::cout << "Idx >= maximum_range !!!" << std::endl;
        idx = maximum_range - 1;
    }
}

template <class DATA_TYPE>
FUNC_T typename Maps_keyframe<DATA_TYPE>::Mapping_cell_ptr Maps_keyframe<DATA_TYPE>::find_cell( const PT_TYPE &pt, int if_add )
{
    Map_pt_cell_it it = m_map_pt_cell.find( pt );
    if ( it == m_map_pt_cell.end() )
    {
        return nullptr;
    }
    else
    {
        return it->second;
    }
}

template <class DATA_TYPE>
FUNC_T void Maps_keyframe<DATA_TYPE>::update_features_of_each_cells( int if_recompute )
{
    if ( m_set_cell.size() != m_last_update_feature_cells_number )
    {
        for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
        {
            ( *it )->determine_feature( if_recompute );
        }
    }
    m_last_update_feature_cells_number = m_set_cell.size();
}

template <class DATA_TYPE>
FUNC_T void Maps_keyframe<DATA_TYPE>::add_cells( const std::set<Mapping_cell_ptr> &cells_vec )
{
    unsigned int last_cell_numbers = 0;
    for ( typename std::set<Mapping_cell_ptr>::iterator it = cells_vec.begin(); it != cells_vec.end(); it++ )
    {
        m_set_cell.insert( *( it ) );
        m_map_pt_cell.insert( std::make_pair( ( *( it ) )->get_center(), *it ) );
        if ( last_cell_numbers != m_set_cell.size() ) // New different cell
        {
            if ( last_cell_numbers == 0 )
            {
                // Init octree
                m_pcl_cells_center->push_back( pcl::PointXYZ( ( *it )->m_center( 0 ), ( *it )->m_center( 1 ), ( *it )->m_center( 2 ) ) );
            }
            last_cell_numbers = m_set_cell.size();
        }
    };
    m_accumulate_frames++;
}

template <class DATA_TYPE>
FUNC_T pcl::PointCloud<pcl_pt> Maps_keyframe<DATA_TYPE>::extract_specify_points( Feature_type select_type )
{
    pcl::PointCloud<pcl_pt> res_pt;
    int                     cell_is_selected_type = 0;
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
    {
        if ( ( *it )->m_feature_type == select_type )
        {
            cell_is_selected_type++;
            res_pt += ( *it )->get_pointcloud();
        }
    }

    // screen_out << "Type is " << ( int ) ( select_type ) << ", total is "<< m_set_cell.size()
    //            << ", cell of type: " << cell_is_selected_type << " , size of pts is " << res_pt.points.size() << std::endl;

    return res_pt;
}

template <class DATA_TYPE>
FUNC_T pcl::PointCloud<pcl_pt> Maps_keyframe<DATA_TYPE>::get_all_pointcloud()
{
    pcl::PointCloud<pcl_pt> res_pt;
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
    {
        res_pt += ( *it )->get_pointcloud();
    }
    return res_pt;
}

template <class DATA_TYPE>
FUNC_T typename Maps_keyframe<DATA_TYPE>::PT_TYPE Maps_keyframe<DATA_TYPE>::get_center()
{
    PT_TYPE cell_center;
    cell_center.setZero();
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
    {
        cell_center += ( *( it ) )->get_center();
    }
    cell_center *= ( 1.0 / ( float ) m_set_cell.size() );
    return cell_center;
}

template <class DATA_TYPE>
FUNC_T float Maps_keyframe<DATA_TYPE>::get_ratio_range_of_cell( typename Maps_keyframe<DATA_TYPE>::PT_TYPE &cell_center, float ratio, std::vector<PT_TYPE> *err_vec )
{
    cell_center = get_center();
    std::set<float> dis_vec;
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
    {
        PT_TYPE dis = ( *( it ) )->get_center() - cell_center;
        if ( err_vec != nullptr )
        {
            err_vec->push_back( dis );
        }
        dis_vec.insert( ( float ) dis.norm() );
    }
    // https://stackoverflow.com/questions/1033089/can-i-increment-an-iterator-by-just-adding-a-number
    return *std::next( dis_vec.begin(), std::ceil( ( dis_vec.size() - 1 ) * ratio ) );
}

template <class DATA_TYPE>
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Maps_keyframe<DATA_TYPE>::apply_guassian_blur( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &in_mat,
                                                                                                    int                                                         kernel_size,
                                                                                                    float                                                       sigma )
{
    cv::Size                                             kernel = cv::Size( 2 * kernel_size + 1, 2 * kernel_size + 1 );
    cv::Mat                                              cv_img;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> res_img;
    res_img = add_padding_to_feature_image( in_mat, kernel_size + 1, kernel_size + 1 );
    cv::eigen2cv( res_img, cv_img );
    cv::GaussianBlur( cv_img, cv_img, kernel, sigma );
    cv::cv2eigen( cv_img, res_img );
    return res_img.block( kernel_size + 1, kernel_size + 1, in_mat.rows(), in_mat.cols() );
}

template <class DATA_TYPE>
std::string Maps_keyframe<DATA_TYPE>::get_frame_info()
{
    char              ss_str[ 10000 ] = "";
    std::stringstream ss( ss_str );
    ss << "===== Frame info =====" << std::endl;
    ss << "Total cell numbers: " << m_set_cell.size() << std::endl;
    ss << "Num_frames: " << m_accumulate_frames << std::endl;
    ss << "Line_cell_all: " << m_feature_vecs_line.size() << std::endl;
    ss << "Plane_cell_all: " << m_feature_vecs_plane.size() << std::endl;
    ss << "==========" << std::endl;
    return ss.str();
}

template <class DATA_TYPE>
FUNC_T void Maps_keyframe<DATA_TYPE>::generate_feature_img( std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> &         feature_vecs_line,
                                                            std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> &         feature_vecs_plane,
                                                            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line,
                                                            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane )
{
    Eigen::Matrix<DATA_TYPE, 3, 3> eigen_vector;
    Eigen::Matrix<DATA_TYPE, 3, 1> eigen_val;

    eigen_decompose_of_featurevector( feature_vecs_plane, eigen_vector, eigen_val );
    eigen_vector = eigen_vector.rowwise().reverse().eval();
    eigen_val = eigen_val.colwise().reverse().eval();
    eigen_vector.col( 2 ) = eigen_vector.col( 0 ).cross( eigen_vector.col( 1 ) );
    //screen_out << "Eigen value = " << eigen_val.transpose() << std::endl;

    feature_img_line.resize( PHI_RESOLUTION, THETA_RESOLUTION );
    feature_img_plane.resize( PHI_RESOLUTION, THETA_RESOLUTION );

    feature_img_line.setZero();
    feature_img_plane.setZero();

    int                            theta_idx = 0;
    int                            beta_idx = 0;
    Eigen::Matrix<DATA_TYPE, 3, 1> affined_vector;

    for ( size_t i = 0; i < feature_vecs_plane.size(); i++ )
    {
        affined_vector = eigen_vector.transpose() * feature_vecs_plane[ i ];
        feature_direction( affined_vector, theta_idx, beta_idx );
        feature_img_plane( theta_idx, beta_idx ) = feature_img_plane( theta_idx, beta_idx ) + 1;
    }

    for ( size_t i = 0; i < feature_vecs_line.size(); i++ )
    {
        affined_vector = eigen_vector.transpose() * feature_vecs_line[ i ];
        feature_direction( affined_vector, theta_idx, beta_idx );
        feature_img_line( theta_idx, beta_idx ) = feature_img_line( theta_idx, beta_idx ) + 1;
    }

    m_ratio_nonzero_line = ratio_of_nonzero_in_img( feature_img_line );
    m_ratio_nonzero_plane = ratio_of_nonzero_in_img( feature_img_plane );
    feature_img_line = apply_guassian_blur( feature_img_line, 4, 4 );
    feature_img_plane = apply_guassian_blur( feature_img_plane, 4, 4 );
}

template <class DATA_TYPE>
FUNC_T void Maps_keyframe<DATA_TYPE>::extract_feature_mapping_new( std::set<Mapping_cell_ptr>                            cell_vec,
                                                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line,
                                                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane,
                                                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line_roi,
                                                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane_roi,
                                                                   int                                                   if_recompute )
{
    update_features_of_each_cells( if_recompute );
    float       ratio = 0.90;
    Eigen_Point keyframe_center;
    m_roi_range = get_ratio_range_of_cell( keyframe_center, ratio );
    m_feature_vecs_plane.clear();
    m_feature_vecs_line.clear();
    m_feature_vecs_plane_roi.clear();
    m_feature_vecs_line_roi.clear();

    Map_pt_cell_it it;
    m_feature_vecs_plane.reserve( cell_vec.size() );
    m_feature_vecs_line.reserve( cell_vec.size() );
    m_feature_vecs_plane_roi.reserve( cell_vec.size() );
    m_feature_vecs_line_roi.reserve( cell_vec.size() );

    for ( auto it = cell_vec.begin(); it != cell_vec.end(); it++ )
    {
        Mapping_cell_ptr cell = *it;

        auto feature_type = cell->m_feature_type;
        if ( feature_type == Feature_type::e_feature_line )
        {
            m_feature_vecs_line.push_back( cell->m_feature_vector.template cast<DATA_TYPE>() );
        }
        if ( feature_type == Feature_type::e_feature_plane )
        {
            m_feature_vecs_plane.push_back( cell->m_feature_vector.template cast<DATA_TYPE>() );
        }
        if ( ( cell->get_center() - keyframe_center ).norm() < m_roi_range )
        {
            if ( feature_type == Feature_type::e_feature_line )
            {
                m_feature_vecs_line_roi.push_back( cell->m_feature_vector.template cast<DATA_TYPE>() );
            }
            if ( feature_type == Feature_type::e_feature_plane )
            {
                m_feature_vecs_plane_roi.push_back( cell->m_feature_vector.template cast<DATA_TYPE>() );
            }
        }
    }

    //screen_out << get_frame_info() << std::endl;
    screen_out << "New keyframe, total cell numbers: " << cell_vec.size();
    screen_out << ", num_frames: " << m_accumulate_frames;
    screen_out << ", line_cell_all: " << m_feature_vecs_line.size();
    screen_out << ", plane_cell_all: " << m_feature_vecs_plane.size() << std::endl;

    generate_feature_img( m_feature_vecs_line_roi, m_feature_vecs_plane_roi, feature_img_line_roi, feature_img_plane_roi );
    generate_feature_img( m_feature_vecs_line, m_feature_vecs_plane, feature_img_line, feature_img_plane );
};

//ANCHOR keyframe::analyze
template <class DATA_TYPE>
FUNC_T int Maps_keyframe<DATA_TYPE>::analyze( int if_recompute )
{
    update_features_of_each_cells();

    extract_feature_mapping_new( m_set_cell, m_feature_img_line, m_feature_img_plane, m_feature_img_line_roi, m_feature_img_plane_roi, if_recompute );
    return 0;
}

template <class DATA_TYPE>
FUNC_T std::string Maps_keyframe<DATA_TYPE>::to_json_string( int &avail_cell_size )
{
    std::string str;
    str.reserve( m_map_pt_cell.size() * 1e4 );
    std::stringstream str_s( str );
    str_s << "[";
    avail_cell_size = 0;
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); )
    {
        Mapping_cell_ptr cell = *it;

        if ( avail_cell_size != 0 )
        {
            str_s << ",";
        }
        str_s << cell->to_json_string();
        avail_cell_size++;

        it++;
        if ( it == m_set_cell.end() )
        {
            break;
        }
    }
    str_s << "]";
    return str_s.str();
}

template <class DATA_TYPE>
FUNC_T void Maps_keyframe<DATA_TYPE>::save_to_file( const std::string &path, const std::string &file_name )
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
void Maps_keyframe<DATA_TYPE>::save_feature_image_to_json( std::string file_name )
{
    std::fstream ofs;
    ofs.open( file_name.c_str(), std::ios_base::out );
    if ( !ofs.is_open() )
    {
        screen_out << "Open file " << file_name << " fail!!! please check: " << std::endl;
    }
    rapidjson::Document     document;
    rapidjson::StringBuffer sb;
#if IF_JSON_PRETTY_WRITTER
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer( sb );
#else
    rapidjson::Writer<rapidjson::StringBuffer> writer( sb );
#endif
    writer.StartObject();               // Between StartObject()/EndObject(),
    writer.SetMaxDecimalPlaces( 1000 ); // like set_precision
    writer.Key( "Cols" );
    writer.Int( PHI_RESOLUTION );
    writer.Key( "Rows" );
    writer.Int( THETA_RESOLUTION );
    Common_tools::save_mat_to_jason_writter( writer, "Line", m_feature_img_line );
    Common_tools::save_mat_to_jason_writter( writer, "Plane", m_feature_img_plane );
    Common_tools::save_mat_to_jason_writter( writer, "Line_roi", m_feature_img_line_roi );
    Common_tools::save_mat_to_jason_writter( writer, "Plane_roi", m_feature_img_plane_roi );
    writer.EndObject();
    screen_out << "Save feature images to " << file_name << std::endl;
    ofs << std::string( sb.GetString() ) << std::endl;
    ofs.close();
}

template <class DATA_TYPE>
void Maps_keyframe<DATA_TYPE>::display()
{
    m_if_verbose_screen_printf = 0;
    for ( auto it = m_set_cell.begin(); it != m_set_cell.end(); it++ )
    {
        screen_out << "Center of cell is: " << ( *( it ) )->get_center().transpose() << std::endl;
    }
    m_if_verbose_screen_printf = 1;
}
