// Author: Jiarong Lin          ziv.lin.ljr@gmail.com
#pragma once
#define USE_HASH 1

#include "common.h"
#include "common_tools.h"
#include "pcl_tools.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>

#include <Eigen/Eigen>
#include <boost/format.hpp>
#include <iomanip>
#include <map>
#include <math.h>
#include <mutex>
#include <thread>
#include <vector>
#if USE_HASH
#include <unordered_map>
#else
#endif

#define IF_COV_INIT_IDENTITY 0
#define IF_EIGEN_REPLACE 1
#define IF_ENABLE_INCREMENTAL_UPDATE_MEAN_COV 0
#define IF_ENABLE_DUMP_PCL_PTS 1
#define IF_JSON_PRETTY_WRITTER 0

// Resolution of 2d histogram
#define PHI_RESOLUTION 60
#define THETA_RESOLUTION 60

typedef pcl::PointXYZI pcl_pt;

// typedef double         COMP_TYPE;
typedef float COMP_TYPE;

#define FUNC_T
// #define FUNC_T inline

enum Feature_type
{
    e_feature_sphere = 0,
    e_feature_line = 1,
    e_feature_plane = 2
};

template <typename DATA_TYPE>
class Points_cloud_cell
{
  public:
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> Eigen_Point;

    typedef Eigen::Matrix<DATA_TYPE, 3, 1> PT_TYPE;
    DATA_TYPE                              m_resolution;
    Eigen::Matrix<DATA_TYPE, 3, 1>         m_center;

    //private:
    Eigen::Matrix<COMP_TYPE, 3, 1> m_xyz_sum;
    Eigen::Matrix<COMP_TYPE, 3, 1> m_mean, m_mean_last;
    Eigen::Matrix<COMP_TYPE, 3, 3> m_cov_mat, m_cov_mat_last, m_cov_mat_avoid_singularity;
    Eigen::Matrix<COMP_TYPE, 3, 3> m_icov_mat;

    /** \brief Eigen vectors of voxel covariance matrix */
    Eigen::Matrix<COMP_TYPE, 3, 3>                m_eigen_vec; // Eigen vector of covariance matrix
    Eigen::Matrix<COMP_TYPE, 3, 1>                m_eigen_val; // Eigen value of covariance values
    int                                           m_last_eigen_decompose_size = 0;
    int                                           m_create_frame_idx = 0;
    int                                           m_last_update_frame_idx = 0;
    Feature_type                                  m_feature_type = e_feature_sphere;
    double                                        m_feature_determine_threshold_line = 1.0 / 3.0;
    double                                        m_feature_determine_threshold_plane = 1.0 / 3.0;
    std::shared_ptr<Points_cloud_cell<DATA_TYPE>> m_previous_visited_cell = nullptr;
    Eigen::Matrix<COMP_TYPE, 3, 1>                m_feature_vector;

  public:
    std::vector<PT_TYPE>        m_points_vec;
    pcl::PointCloud<pcl_pt>     m_pcl_pc_vec;
    DATA_TYPE                   m_cov_det_sqrt;
    bool                        m_mean_need_update = true;
    bool                        m_covmat_need_update = true;
    bool                        m_icovmat_need_update = true;
    size_t                      m_maximum_points_size = ( size_t ) 1e2;
    int                         m_if_incremental_update_mean_and_cov = 0;
    std::shared_ptr<std::mutex> m_mutex_cell;
    double                      m_last_update_time;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    Points_cloud_cell();
    Points_cloud_cell( const PT_TYPE &cell_center, const DATA_TYPE &res = 1.0 );
    ~Points_cloud_cell();

    FUNC_T std::string to_json_string();
    FUNC_T void        save_to_file( const std::string &path = std::string( "./" ), const std::string &file_name = std::string( "" ) );
    void               set_data_need_update( int if_update_sum = 0 );
    FUNC_T int         get_points_count();
    FUNC_T Eigen::Matrix<DATA_TYPE, 3, 1> get_center();
    FUNC_T Eigen::Matrix<DATA_TYPE, 3, 1> get_mean();
    FUNC_T void                           covmat_eig_decompose( int if_force_update = 0 );
    FUNC_T Eigen::Matrix<COMP_TYPE, 3, 3> get_cov_mat_avoid_singularity(); // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
    FUNC_T Eigen::Matrix<DATA_TYPE, 3, 3> get_covmat();
    FUNC_T Eigen::Matrix<DATA_TYPE, 3, 3> get_icovmat();
    FUNC_T pcl::PointCloud<pcl_pt> get_pointcloud();
    FUNC_T pcl::PointCloud<pcl_pt> get_oldest_pointcloud();
    FUNC_T std::vector<PT_TYPE> get_pointcloud_eigen();
    FUNC_T void                 set_pointcloud( pcl::PointCloud<pcl_pt> &pc_in );

    FUNC_T void clear_data();
    FUNC_T void append_pt( const PT_TYPE &pt );
    FUNC_T void set_target_pc( const std::vector<PT_TYPE> &pt_vec );
    FUNC_T Feature_type determine_feature( int if_recompute = 0 );
};

template <typename DATA_TYPE>
class Points_cloud_map
{
  public:
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> PT_TYPE;
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> Eigen_Point;
    typedef Points_cloud_cell<DATA_TYPE>   Mapping_cell;
    typedef std::shared_ptr<Mapping_cell>  Mapping_cell_ptr;
#if USE_HASH
    typedef std::unordered_map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare>                    Map_pt_cell;
    typedef typename std::unordered_map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare>::iterator Map_pt_cell_it;
#else
    typedef std::map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Pt_compare>                    Map_pt_cell;
    typedef typename std::map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Pt_compare>::iterator Map_pt_cell_it;
#endif
    DATA_TYPE                     m_x_min, m_x_max;
    DATA_TYPE                     m_y_min, m_y_max;
    DATA_TYPE                     m_z_min, m_z_max;
    DATA_TYPE                     m_resolution; // resolution mean the distance of a cute to its bound.
    Common_tools::Timer           m_timer;
    std::set<Mapping_cell_ptr>    m_cell_vec;
    int                           m_if_incremental_update_mean_and_cov = IF_ENABLE_INCREMENTAL_UPDATE_MEAN_COV;
    //std::unique_ptr< std::mutex >             m_mapping_mutex;
    std::shared_ptr<std::mutex>             m_mapping_mutex;
    std::shared_ptr<std::mutex>             m_octotree_mutex;
    std::shared_ptr<std::mutex>             m_mutex_addcell;
    std::string                             m_json_file_name;
    std::vector<std::set<Mapping_cell_ptr>> m_frame_with_cell_index;
    float                                   m_ratio_nonzero_line, m_ratio_nonzero_plane;
    int                                     m_current_frame_idx;
    int                                     m_minimum_revisit_threshold = std::numeric_limits<int>::max();

    Map_pt_cell    m_map_pt_cell; // using hash_map
    Map_pt_cell_it m_map_pt_cell_it;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> m_octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>( 0.0001 );
    pcl::PointCloud<pcl::PointXYZ>::Ptr                m_pcl_cells_center = pcl::PointCloud<pcl::PointXYZ>::Ptr( new pcl::PointCloud<pcl::PointXYZ>() );
    int                                                m_initialized = false;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    Points_cloud_map();
    ~Points_cloud_map();

    FUNC_T void set_update_mean_and_cov_incrementally( int flag = 1 );

    FUNC_T int get_cells_size();

    FUNC_T PT_TYPE find_cell_center( const PT_TYPE &pt );

    FUNC_T void clear_data();

    FUNC_T void set_point_cloud( const std::vector<PT_TYPE> &             input_pt_vec,
                                 std::set<std::shared_ptr<Mapping_cell>> *cell_vec = nullptr,  int if_octree = 1 );

    FUNC_T void append_cloud( const std::vector<PT_TYPE> &input_pt_vec, std::set<Mapping_cell_ptr> *cell_vec = nullptr );

    template <typename T>
    FUNC_T void set_resolution( T resolution )
    {
        m_resolution = DATA_TYPE( resolution * 0.5 );
        screen_out << "Resolution is set as: " << m_resolution << std::endl;
        m_octree.setResolution( m_resolution );
    };

    FUNC_T DATA_TYPE get_resolution();

    FUNC_T Mapping_cell_ptr add_cell( const PT_TYPE &cell_center );

    FUNC_T Mapping_cell_ptr find_cell( const PT_TYPE &pt, int if_add = 1, int if_treat_revisit = 0 );

    template <typename T>
    FUNC_T std::vector<Mapping_cell_ptr> find_cells_in_radius( T pt, float searchRadius = 0 )
    {
        // std::unique_lock< std::mutex > lock( *m_octotree_mutex );

        std::vector<Mapping_cell_ptr> cells_vec;
        pcl::PointXYZ                 searchPoint = PCL_TOOLS::eigen_to_pcl_pt<pcl::PointXYZ>( pt );
        std::vector<int>              cloudNWRSearch;
        std::vector<float>            cloudNWRRadius;

        if ( searchRadius == 0 )
        {
            m_octree.radiusSearch( searchPoint, m_resolution, cloudNWRSearch, cloudNWRRadius );
        }
        else
        {
            m_octree.radiusSearch( searchPoint, searchRadius, cloudNWRSearch, cloudNWRRadius );
        }

        PT_TYPE eigen_pt;
        for ( size_t i = 0; i < cloudNWRSearch.size(); i++ )
        {

            eigen_pt = PCL_TOOLS::pcl_pt_to_eigen<DATA_TYPE>( m_octree.getInputCloud()->points[ cloudNWRSearch[ i ] ] );
            cells_vec.push_back( find_cell( eigen_pt ) );
        }

        return cells_vec;
    };

    FUNC_T std::string to_json_string( int &avail_cell_size = 0 );

    FUNC_T void save_to_file( const std::string &path = std::string( "./" ),
                              const std::string &file_name = std::string( "" ) );
    template <typename T>
    FUNC_T T *get_json_array( const rapidjson::Document::Array &json_array )
    {
        T *res_mat = new T[ json_array.Size() ];
        for ( size_t i = 0; i < json_array.Size(); i++ )
        {
            res_mat[ i ] = ( T ) json_array[ i ].GetDouble();
        }
        return res_mat;
    }

    FUNC_T int load_mapping_from_file( const std::string &file_name = std::string( "./mapping.json" ) );

    FUNC_T std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> load_pts_from_file( const std::string &file_name = std::string( "./mapping.json" ) );
    //template <typename T>
    FUNC_T std::vector<PT_TYPE> query_point_cloud( std::vector<Mapping_cell *> &cell_vec );
    FUNC_T pcl::PointCloud<pcl_pt> extract_specify_points( Feature_type select_type );
    FUNC_T pcl::PointCloud<pcl_pt> get_all_pointcloud();
};

template <typename DATA_TYPE>
class Maps_keyframe
{
  public:
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> PT_TYPE;
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> Eigen_Point;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double m_pose_buffer[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };

    Eigen::Map<Eigen::Quaterniond> m_pose_q = Eigen::Map<Eigen::Quaterniond>( m_pose_buffer );
    Eigen::Map<Eigen::Vector3d>    m_pose_t = Eigen::Map<Eigen::Vector3d>( m_pose_buffer + 4 );

    typedef Points_cloud_cell<DATA_TYPE>  Mapping_cell;
    typedef std::shared_ptr<Mapping_cell> Mapping_cell_ptr;
#if USE_HASH
    typedef std::unordered_map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare>                    Map_pt_cell;
    typedef typename std::unordered_map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare>::iterator Map_pt_cell_it;
#else
    typedef std::map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Pt_compare>                    Map_pt_cell;
    typedef typename std::map<PT_TYPE, Mapping_cell_ptr, PCL_TOOLS::Pt_compare>::iterator Map_pt_cell_it;
#endif
    Map_pt_cell                         m_map_pt_cell; // using hash_map
    Map_pt_cell_it                      m_map_pt_cell_it;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_pcl_cells_center = pcl::PointCloud<pcl::PointXYZ>::Ptr( new pcl::PointCloud<pcl::PointXYZ>() );

    // ANCHOR cell:resolution
    int                                                  scale = 10;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m_feature_img_line, m_feature_img_plane;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m_feature_img_line_roi, m_feature_img_plane_roi;

    std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> m_feature_vecs_plane, m_feature_vecs_line;
    std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> m_feature_vecs_plane_roi, m_feature_vecs_line_roi;

    Eigen::Matrix<float, 3, 3> m_eigen_R, m_eigen_R_roi;
    float                      m_ratio_nonzero_line, m_ratio_nonzero_plane;
    ADD_SCREEN_PRINTF_OUT_METHOD;
    float                      m_roi_range;
    pcl::PointCloud<PointType> m_accumulated_point_cloud_full;    // The full pointcloud sampled from current frame to last frame
    pcl::PointCloud<PointType> m_accumulated_point_cloud_corners; // The corners sampled from current frame to last frame
    pcl::PointCloud<PointType> m_accumulated_point_cloud_surface; // The surface sampled from current frame to last frame
    std::set<Mapping_cell_ptr> m_set_cell;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> m_octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>( 0.0001 );
    unsigned int                                       m_last_update_feature_cells_number;
    unsigned int                                       m_accumulate_frames = 0;
    pcl::PointCloud<pcl_pt>                            m_pcl_snap_shot_line, m_pcl_snap_shot_plane;
    int                                                m_ending_frame_idx;
    std::shared_ptr<std::mutex>                        m_keyframe_idx;
    Maps_keyframe();

    ~Maps_keyframe();

    void make_index_in_matrix_range( int &idx, int maximum_range );

    //ANCHOR cell::feature_direction
    template <typename T>
    FUNC_T void feature_direction( Eigen::Matrix<T, 3, 1> &vec_3d, int &phi_idx, int &theta_idx )
    {
        if ( vec_3d[ 0 ] < 0 )
        {
            vec_3d *= ( -1.0 );
        }
        int    phi_res = PHI_RESOLUTION;
        int    theta_res = THETA_RESOLUTION;
        double phi_step = M_PI / phi_res;
        double theta_step = M_PI / theta_res;
        double phi = atan2( vec_3d[ 1 ], vec_3d[ 0 ] ) + M_PI / 2;
        double theta = asin( vec_3d[ 2 ] ) + M_PI / 2;

        phi_idx = ( std::floor( phi / phi_step ) );
        theta_idx = ( std::floor( theta / theta_step ) );
        make_index_in_matrix_range( phi_idx, PHI_RESOLUTION );
        make_index_in_matrix_range( theta_idx, THETA_RESOLUTION );
    }

    FUNC_T Mapping_cell_ptr find_cell( const PT_TYPE &pt, int if_add = 1 );

    template <typename T>
    FUNC_T std::vector<Mapping_cell_ptr> find_cells_in_radius( T pt, float searchRadius )
    {
        std::vector<Mapping_cell_ptr> cells_vec;
        pcl::PointXYZ                 searchPoint = PCL_TOOLS::eigen_to_pcl_pt<pcl::PointXYZ>( pt );
        std::vector<int>              cloudNWRSearch;
        std::vector<float>            cloudNWRRadius;
        printf_line;

        m_octree.radiusSearch( searchPoint, searchRadius, cloudNWRSearch, cloudNWRRadius );

        PT_TYPE eigen_pt;
        for ( size_t i = 0; i < cloudNWRSearch.size(); i++ )
        {
            eigen_pt = PCL_TOOLS::pcl_pt_to_eigen<DATA_TYPE>( m_octree.getInputCloud()->points[ cloudNWRSearch[ i ] ] );
            cells_vec.push_back( find_cell( eigen_pt ) );
        }

        return cells_vec;
    }

    template <typename T>
    FUNC_T static void refine_feature_img( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &feature_img )
    {
        int rows = feature_img.rows();
        int cols = feature_img.cols();
        //printf( "Rows = %d, cols = %d\r\n", rows, cols );
        if ( feature_img.row( 0 ).maxCoeff() < feature_img.row( rows - 1 ).maxCoeff() )
        {
            feature_img = feature_img.colwise().reverse().eval();
        }

        if ( ( feature_img.block( 0, 0, 2, round( cols / 2 ) ) ).maxCoeff() < ( feature_img.block( 0, round( cols / 2 ), 2, round( cols / 2 ) ) ).maxCoeff() )
        {
            feature_img = feature_img.rowwise().reverse().eval();
        }
    }

    FUNC_T static float ratio_of_nonzero_in_img( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img )
    {
        int count = 0;
        for ( int i = 0; i < img.rows(); i++ )
        {
            for ( int j = 0; j < img.cols(); j++ )
                if ( img( i, j ) >= 1.0 )
                    count++;
        }
        return ( float ) ( count ) / ( img.rows() * img.cols() );
    }

    // ANCHOR keyframe::max_similiarity_of_two_image
    FUNC_T static float max_similiarity_of_two_image( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b, float minimum_zero_ratio = 0.00 )
    {
        if ( ratio_of_nonzero_in_img( img_a ) < minimum_zero_ratio )
        {
            return 0;
        }

        if ( ratio_of_nonzero_in_img( img_b ) < minimum_zero_ratio )
        {
            return 0;
        }
        size_t cols = img_a.cols();
        size_t rows = img_a.rows();
        float  max_res = -0;

        cv::Mat                                              hist_a, hist_b;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi;
        img_b_roi.resize( rows, cols );
        float res = 0;
        // img_b_roi = img_b;
        if ( 0 )
        {
            for ( size_t i = 0; i < rows * 1.0; i++ )
            {
                //Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_b_roi = img_b.block( i, 0, ( int ) std::round( rows / 2 ), cols );
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi_up = img_b.block( i, 0, rows - i, cols );
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi_down = img_b.block( 0, 0, i, cols );

                img_b_roi << img_b_roi_up, img_b_roi_down;

                res = similiarity_of_two_image_opencv( img_a, img_b_roi );

                if ( fabs( res ) > fabs( max_res ) )
                    max_res = fabs( res );
            }
        }
        else
        {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> res_img;
            res_img = add_padding_to_feature_image( img_b, PHI_RESOLUTION * 0.5, THETA_RESOLUTION * 0.5 );
            max_res = max_similiarity_of_two_image_opencv( img_a, res_img );
        }

        //std::cout << hist_a + hist_b << std::endl;
        return max_res;
    }

    FUNC_T static float similiarity_of_two_image_opencv( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b, int method = CV_COMP_CORREL )
    {
        cv::Mat hist_a, hist_b;
        cv::eigen2cv( img_a, hist_a );
        cv::eigen2cv( img_b, hist_b );
        return cv::compareHist( hist_a, hist_b, method ); // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
    }

    FUNC_T static float max_similiarity_of_two_image_opencv( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b, int method = CV_COMP_CORREL )
    {
        cv::Mat                                              hist_a, hist_b;
        int                                                  cols = img_a.cols();
        int                                                  rows = img_a.rows();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_a_roi = img_a.block( 0, 0, rows, cols );
        cv::eigen2cv( img_a_roi, hist_a );
        cv::eigen2cv( img_b, hist_b );
        cv::Mat result;
        cv::matchTemplate( hist_b, hist_a, result, CV_TM_CCORR_NORMED );
        double    minVal;
        double    maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
        return maxVal;
        //return cv::compareHist( hist_a, hist_b, method ); // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
    }

    FUNC_T void update_features_of_each_cells( int if_recompute = 0 );
    FUNC_T void add_cells( const std::set<Mapping_cell_ptr> &cells_vec );

    FUNC_T pcl::PointCloud<pcl_pt> extract_specify_points( Feature_type select_type );

    FUNC_T pcl::PointCloud<pcl_pt> get_all_pointcloud();

    FUNC_T PT_TYPE get_center();

    FUNC_T float get_ratio_range_of_cell( PT_TYPE &cell_center = PT_TYPE( 0, 0, 0 ), float ratio = 0.8, std::vector<PT_TYPE> *err_vec = nullptr );

    FUNC_T static Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> add_padding_to_feature_image( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &in_mat,
                                                                                                     int                                                         padding_size_x,
                                                                                                     int                                                         padding_size_y )
    {

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> res_img;
        int                                                  size_w = in_mat.cols();
        int                                                  size_h = in_mat.rows();

        res_img.resize( size_h + 2 * padding_size_y, size_w + 2 * padding_size_x );
        // res_img.setZero();
        res_img.block( padding_size_y, padding_size_x, size_h, size_w ) = in_mat;

        // Padding four corners

        // Top left
        res_img.block( 0, 0, padding_size_y, padding_size_x ) = in_mat.block( size_h - padding_size_y, size_w - padding_size_x, padding_size_y, padding_size_x );
        // Top right
        res_img.block( 0, size_w + padding_size_x, padding_size_y, padding_size_x ) = in_mat.block( size_h - padding_size_y, 0, padding_size_y, padding_size_x );
        // Bottom left
        res_img.block( size_h + padding_size_y, 0, padding_size_y, padding_size_x ) = in_mat.block( 0, size_w - padding_size_x, padding_size_y, padding_size_x );
        // Bottom right
        res_img.block( size_h + padding_size_y, size_w + padding_size_x, padding_size_y, padding_size_x ) = in_mat.block( 0, 0, padding_size_y, padding_size_x );

        // Padding four blocks
        // Up
        res_img.block( 0, padding_size_x, padding_size_y, size_w ) = in_mat.block( size_h - padding_size_y, 0, padding_size_y, size_w );
        // Down
        res_img.block( size_h + padding_size_y, padding_size_x, padding_size_y, size_w ) = in_mat.block( 0, 0, padding_size_y, size_w );
        // Left
        res_img.block( padding_size_y, 0, size_h, padding_size_x ) = in_mat.block( 0, size_w - padding_size_x, size_h, padding_size_x );
        // Right
        res_img.block( padding_size_y, size_w + padding_size_x, size_h, padding_size_x ) = in_mat.block( 0, 0, size_h, padding_size_x );

        return res_img;
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> apply_guassian_blur( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &in_mat,
                                                                              int                                                         kernel_size,
                                                                              float                                                       sigma );

    std::string get_frame_info();

    FUNC_T void generate_feature_img( std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> &         feature_vecs_line,
                                      std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> &         feature_vecs_plane,
                                      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line,
                                      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane );

    FUNC_T void extract_feature_mapping_new( std::set<Mapping_cell_ptr>                            cell_vec,
                                             Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line,
                                             Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane,
                                             Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line_roi,
                                             Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane_roi,
                                             int                                                   if_recompute = 0 );
    //ANCHOR keyframe::analyze
    FUNC_T int analyze( int if_recompute = 0 );

    FUNC_T std::string to_json_string( int &avail_cell_size = 0 );

    FUNC_T void save_to_file( const std::string &path = std::string( "./" ), const std::string &file_name = std::string( "" ) );
    
    template <typename T>
    FUNC_T void eigen_decompose_of_featurevector( std::vector<Eigen::Matrix<T, 3, 1>> &feature_vectors, Eigen::Matrix<T, 3, 3> &eigen_vector, Eigen::Matrix<T, 3, 1> &eigen_val )
    {
        Eigen::Matrix<double, 3, 3> mat_cov;
        mat_cov.setIdentity();
        // mat_cov.setZero();
        for ( size_t i = 0; i < feature_vectors.size(); i++ )
        {
            mat_cov = mat_cov + ( feature_vectors[ i ] * feature_vectors[ i ].transpose() ).template cast<double>();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eigensolver;
        eigensolver.compute( mat_cov );
        eigen_val = eigensolver.eigenvalues().template cast<T>();
        eigen_vector = eigensolver.eigenvectors().template cast<T>();
    }

    FUNC_T static float similiarity_of_two_image( const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b )
    {
        assert( ( ( img_a.rows() == img_b.rows() ) && ( img_a.cols() == img_b.cols() ) ) );

        auto img_sub_mea_a = img_a.array() - img_a.mean();
        auto img_sub_mea_b = img_b.array() - img_b.mean();

        float product = ( ( img_sub_mea_a ).cwiseProduct( img_sub_mea_b ) ).mean();
        int   devide_size = img_a.rows() * img_a.cols() - 1;
        float std_a = ( img_sub_mea_a.array().pow( 2 ) ).sum() / devide_size;
        float std_b = ( img_sub_mea_b.array().pow( 2 ) ).sum() / devide_size;
        return sqrt( product * product / std_a / std_b );
    };

    void save_feature_image_to_json( std::string file_name );
    void display();
};
