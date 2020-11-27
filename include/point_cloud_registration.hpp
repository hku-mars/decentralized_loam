// Author: Jiarong Lin          ziv.lin.ljr@gmail.com

#ifndef POINT_CLOUD_REGISTRATION_HPP
#define POINT_CLOUD_REGISTRATION_HPP
#include "common.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <future>
#include <iostream>
#include <math.h>
#include <mutex>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "cell_map_keyframe.hpp"
#include "ceres_icp.hpp"
#include "common.h"
#include "pcl_tools.hpp"
#include "tools_ceres.hpp"
#include "tools_logger.hpp"
#include "tools_timer.hpp"

#define CORNER_MIN_MAP_NUM 0
#define SURFACE_MIN_MAP_NUM 50

#define BLUR_SCALE 1.0

using PCL_TOOLS::pcl_pt_to_eigend;
using std::cout;
using std::endl;

class Point_cloud_registration
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //ceres::LinearSolverType solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::LinearSolverType solver_type = ceres::DENSE_QR; // SPARSE_NORMAL_CHOLESKY | DENSE_QR | DENSE_SCHUR

    int    line_search_num = 5;
    int    IF_LINE_FEATURE_CHECK = 1;
    int    plane_search_num = 8;
    int    IF_PLANE_FEATURE_CHECK = 1;
    int    ICP_PLANE = 1;
    int    ICP_LINE = 1;
    double m_para_buffer_RT[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    double m_para_buffer_RT_last[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    double m_para_buffer_incremental[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };

    std::vector< vec_3, Eigen::aligned_allocator<vec_3>  > m_cst_plane_vec, m_cst_line_vec;
    Eigen::Map<Eigen::Quaterniond> m_q_w_incre = Eigen::Map<Eigen::Quaterniond>( m_para_buffer_incremental );
    Eigen::Map<Eigen::Vector3d>    m_t_w_incre = Eigen::Map<Eigen::Vector3d>( m_para_buffer_incremental + 4 );

    double m_interpolatation_theta;

    int m_if_motion_deblur = 0;

    double                      m_angular_diff = 0;
    double                      m_t_diff = 0;
    double                      m_maximum_dis_plane_for_match = 50.0;
    double                      m_maximum_dis_line_for_match = 2.0;
    Eigen::Matrix<double, 3, 1>                   m_interpolatation_omega;
    Eigen::Matrix<double, 3, 3>                   m_interpolatation_omega_hat;
    Eigen::Matrix<double, 3, 3>                   m_interpolatation_omega_hat_sq2;
    mat_66_noalign                                m_cov_mat;
    vec_6                                         m_cov_mat_eig_vec;
    const Eigen::Quaterniond m_q_I = Eigen::Quaterniond( 1, 0, 0, 0 );

    pcl::KdTreeFLANN<PointType> m_kdtree_corner_from_map;
    pcl::KdTreeFLANN<PointType> m_kdtree_surf_from_map;

    Eigen::Quaterniond m_q_w_curr, m_q_w_last;
    Eigen::Vector3d    m_t_w_curr, m_t_w_last;

    Common_tools::File_logger *m_logger_common;
    Common_tools::File_logger *m_logger_pcd;
    Common_tools::File_logger *m_logger_timer;

    Common_tools::Timer *m_timer;
    int                  m_current_frame_index;
    int                  m_mapping_init_accumulate_frames = 100;
    float                m_last_time_stamp = 0;
    float                m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float                m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float                m_max_final_cost = 100.0;
    int                  m_para_icp_max_iterations = 20;
    int                  m_para_cere_max_iterations = 100;
    int                  m_para_cere_prerun_times = 2;
    int                  m_maximum_degenerate_direction = 2;
    float                m_minimum_pt_time_stamp = 0;
    float                m_maximum_pt_time_stamp = 1.0;
    double               m_minimum_icp_R_diff = 0.01;
    double               m_minimum_icp_T_diff = 0.01;
    double               m_degenerate_threshold = 1e-6;
    double               m_inliner_dis = 0.02;
    double               m_inlier_ratio = 0.80;

    double                                      m_inlier_threshold;
    ceres::Solver::Summary                      summary;
    ceres::Solver::Summary                      m_final_opt_summary;
    int                                         m_maximum_allow_residual_block = 1e5;
    int                                         m_if_degenerate = 1;
    Common_tools::Random_generator_float<float> m_rand_float;
    ~Point_cloud_registration();

    Point_cloud_registration();

    ADD_SCREEN_PRINTF_OUT_METHOD;
    void   update_transform();
    void   reset_incremental_parameter();
    float  refine_blur( float in_blur, const float &min_blur, const float &max_blur );
    void   set_ceres_solver_bound( ceres::Problem &problem, double *para_buffer_RT );
    double compute_inlier_residual_threshold( std::vector<double> residuals, double ratio );
    void   my_iterate( ceres::Problem *problem, int solver_type = 0,
                       int                              max_iteration = 5,
                       std::vector<Eigen::Quaterniond> *q_vec = nullptr,
                       std::vector<Eigen::Vector3d> *   t_vec = nullptr );

    int find_out_incremental_transfrom( pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map,
                                        pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,
                                        pcl::KdTreeFLANN<PointType> &   kdtree_corner_from_map,
                                        pcl::KdTreeFLANN<PointType> &   kdtree_surf_from_map,
                                        pcl::PointCloud<PointType>::Ptr laserCloudCornerStack,
                                        pcl::PointCloud<PointType>::Ptr laserCloudSurfStack );

    int          find_out_incremental_transfrom( pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map,
                                                 pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,
                                                 pcl::PointCloud<PointType>::Ptr laserCloudCornerStack,
                                                 pcl::PointCloud<PointType>::Ptr laserCloudSurfStack );
    void         compute_interpolatation_rodrigue( const Eigen::Quaterniond &q_in, Eigen::Matrix<double, 3, 1> &angle_axis, double &angle_theta, Eigen::Matrix<double, 3, 3> &hat );
    void         pointAssociateToMap( PointType const *const pi, PointType *const po,
                                      double interpolate_s = 1.0, int if_undistore = 0 );
    void         pointAssociateTobeMapped( PointType const *const pi, PointType *const po );
    unsigned int pointcloudAssociateToMap( pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out,
                                           int if_undistore = 0 );
    unsigned int pointcloudAssociateTbeoMapped( pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out );
};

#endif // POINT_CLOUD_REGISTRATION_HPP
