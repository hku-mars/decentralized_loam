
#pragma once
#include "common_tools.h"
#include <Eigen/Eigen>
#include <array>
#include <map>
#include <mutex>
#include "common.h"

using namespace std;

class Lidar_agent
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int m_lidar_id;
    int m_last_trajectory_idx = 0;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> m_trajectory_R;
    std::vector<vec_3, Eigen::aligned_allocator<vec_3>>                           m_trajectory_T;
    std::vector<mat_66, Eigen::aligned_allocator<mat_66>>                         m_trajectory_cov_mat, m_last_trajectory_cov_mat_res;
    std::vector<double>                                                           m_trajectory_time;
    std::vector<vec_6, Eigen::aligned_allocator<vec_6>>                           m_eig_val_vec;
    vec_6                                                                         m_last_cov_eig_vec;
    Common_tools::Timer                      m_timer;
    Common_tools::File_logger                m_logger;
    double                                   m_odometry_RT_curr[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    double                                   m_odometry_RT_last[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };

    double m_buffer_RT[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    double m_buffer_RT_last[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };
    double m_buffer_RT_last_incre[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };

    Eigen::Map<Eigen::Quaterniond>          m_q_w_curr = Eigen::Map<Eigen::Quaterniond>( m_buffer_RT );
    Eigen::Map<Eigen::Matrix<double, 3, 1>> m_t_w_curr = Eigen::Map<Eigen::Vector3d>( m_buffer_RT + 4 );
    mat_66                                  m_cov_mat_curr, m_cov_mat_last;


    Eigen::Map<Eigen::Quaterniond>          m_q_w_last = Eigen::Map<Eigen::Quaterniond>( m_buffer_RT_last );
    Eigen::Map<Eigen::Matrix<double, 3, 1>> m_t_w_last = Eigen::Map<Eigen::Vector3d>( m_buffer_RT_last + 4 );
    
    vec_6                       m_state_current;
    vec_6                       m_state_last;
    std::shared_ptr<std::mutex> m_mutex;
    ADD_SCREEN_PRINTF_OUT_METHOD;
    
    Lidar_agent();
    ~Lidar_agent();

    void clear_data();
    void init_log( int index = 0 );
    void update_history( double current_time = -1 , int save_log = 1);
    void update( double current_time, const Eigen::Quaterniond &q_curr, const vec_3 &t_curr, const mat_66 &cov_mat, int save_log = 1 );
    void save_trajectory_to_file( const string &file_name, std::vector<double>                                   trajectory_time,
                                  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> &trajectory_R,
                                  std::vector<vec_3, Eigen::aligned_allocator<vec_3>> &                          trajectory_T,
                                  std::vector<mat_66, Eigen::aligned_allocator<mat_66>> &                        trajectory_cov );

    void save_raw_trajectory_to_file( const string &file_name );
    void save_filtered_trajectory_to_file( const string &file_name );
    void print_trajectory();
    void load_from_file( const string &file_name );
};
