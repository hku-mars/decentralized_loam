#pragma once
#include "common_tools.h"
#include "lidar_agent.hpp"
#include "math.h"
#include "tools_ceres.hpp"
#include "tools_logger.hpp"
#include <iostream>
#include <stdio.h>

#include <Eigen/Eigen>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#define ENABLE_PRINTF 0

#define if_mode_delta 1

// using
template <typename T>
void eigen_quaternion_to_rotation_vec( Eigen::Quaternion<T> &q_in, T *r_vec )
{
    // Eigen store quaternion: [qx qy qz qw]
    // Ceres quaternion array: [qw qx qy qz]
    T q[ 4 ];
    q[ 0 ] = q_in.w();
    q[ 1 ] = q_in.x();
    q[ 2 ] = q_in.y();
    q[ 3 ] = q_in.z();
    ceres::QuaternionToAngleAxis<T>( ( T * ) &q, r_vec );
}

template <typename T>
void dump_vector( FILE *fp, const T &vec )
{
    size_t length = vec.size();
    // cout << "Length = "<< length <<endl;

    for ( size_t i = 0; i < length; i++ )
    {
        fprintf( fp, "%lf ", vec( i ) );
        //printf(  "%lf ", vec( i ) );
    }
}

// ANCHOR local ekf
class Local_lidar_ekf
{

  public:
    typedef ceres::Jet<double, 12> Jet_12;
    typedef ceres::Jet<double, 6>  Jet_6;
    Lidar_agent m_lidar_agent;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> m_trajectory_R, m_trajectory_R_filtered;
    std::vector<vec_3, Eigen::aligned_allocator<vec_3>>                           m_trajectory_T, m_trajectory_T_filtered;
    Eigen::Quaterniond m_state_star_R_lastest;
    vec_3              m_state_star_T_lastest;

    std::vector<mat_66, Eigen::aligned_allocator<mat_66>> m_trajectory_cov_mat, m_last_trajectory_cov_mat_res;
    std::vector<double> m_trajectory_time;
    std::vector<vec_6, Eigen::aligned_allocator<vec_6>> m_eig_val_vec;
    vec_12      m_state_X_curr, m_state_bar_x_curr;
    mat_12      m_cov_sigma_curr, m_bar_cov_sigma_curr;
    mat_12_6    m_K_t;
    mat_6_12    m_H_t;
    mat_12      m_covariance_extrinsic; // mat of Sigma_e
    vec_6       m_state_extrinsic;      // vector of extrinsic
    double      m_current_time;
    double      delta_time;
    Eigen::Matrix<Jet_12, 12, 1, Eigen::DontAlign> x_k, x_bar_kplus1, x_hat_kplus1;

    Jet_12 q_k[ 4 ], q_k_inv[ 4 ], bar_q_kplus1[ 4 ];
    FILE * m_fp = nullptr;
    Local_lidar_ekf(){};

    Local_lidar_ekf( Lidar_agent lidar_agent ) : m_lidar_agent( lidar_agent )
    {
        m_trajectory_R = m_lidar_agent.m_trajectory_R;
        m_trajectory_T = m_lidar_agent.m_trajectory_T;
        m_trajectory_cov_mat = m_lidar_agent.m_trajectory_cov_mat;
        m_trajectory_time = m_lidar_agent.m_trajectory_time;
        m_eig_val_vec = m_lidar_agent.m_eig_val_vec;
        m_covariance_extrinsic.setIdentity();
    };

    ~Local_lidar_ekf()
    {
        if ( m_fp != nullptr )
        {
            fflush( m_fp );
            fclose( m_fp );
        }
    };

    void reset( int idx = 0 )
    {
        assert( m_trajectory_T.size() && m_trajectory_R.size() && m_trajectory_time.size() );
        // double r_vec[3], t_vec[3], omega_vec[3], vel_vec[3];
        m_current_time = m_trajectory_time[ idx ];
        vec_12 state;
        eigen_quaternion_to_rotation_vec( m_trajectory_R[ idx ], &state.data()[ 0 ] );
        memcpy( &state.data()[ 3 ], m_trajectory_T[ idx ].data(), 3 * sizeof( double ) );
        memset( &state.data()[ 6 ], 0, 3 * sizeof( double ) );
        memset( &state.data()[ 9 ], 0, 3 * sizeof( double ) );
        m_state_X_curr = state;
        m_cov_sigma_curr = mat_12::Identity() * 1.0;
        cout << "Init state: " << m_state_X_curr.transpose() << endl;
    }

    template <typename T>
    void printf_ceres_vec( T vec, int size )
    {
        cout << std::setprecision( 2 );
        for ( int i = 0; i < size; i++ )
        {
            cout << vec( i ).a << " ";
        }
        cout << endl;
    }

    template <typename T>
    bool ceres_quadternion_delta( const T *x, const T *delta, T *x_plus_delta ) const
    {
        const T squared_norm_delta =
            delta[ 0 ] * delta[ 0 ] + delta[ 1 ] * delta[ 1 ] + delta[ 2 ] * delta[ 2 ];

        T q_delta[ 4 ];
        if ( squared_norm_delta > 0.0 )
        {
            T       norm_delta = sqrt( squared_norm_delta );
            const T sin_delta_by_delta = sin( norm_delta ) / norm_delta;
            q_delta[ 0 ] = cos( norm_delta );
            q_delta[ 1 ] = sin_delta_by_delta * delta[ 0 ];
            q_delta[ 2 ] = sin_delta_by_delta * delta[ 1 ];
            q_delta[ 3 ] = sin_delta_by_delta * delta[ 2 ];
        }
        else
        {
            // We do not just use q_delta = [1,0,0,0] here because that is a
            // constant and when used for automatic differentiation will
            // lead to a zero derivative. Instead we take a first order
            // approximation and evaluate it at zero.
            q_delta[ 0 ] = T( 1.0 );
            q_delta[ 1 ] = delta[ 0 ];
            q_delta[ 2 ] = delta[ 1 ];
            q_delta[ 3 ] = delta[ 2 ];
        }

        ceres::QuaternionProduct( x, q_delta, x_plus_delta );
        return true;
    }

    template <typename T>
    Eigen::Matrix<T, 3, 1> ceres_quadternion_parameterization( const Eigen::Quaternion<T> &q_in )
    {
        Eigen::Matrix<T, 3, 1> res;
        T mod_delta = acos( q_in.w() );
        if ( abs( q_in.w() - 1 ) <= 1e-5 )
        {
            // We do not just use q_delta = [0,0,0] here because that is a
            // constant and when used for automatic differentiation will
            // lead to a zero derivative. Instead we take a first order
            // approximation and evaluate it at zero.
            res( 0 ) = T( q_in.x() );
            res( 1 ) = T( q_in.y() );
            res( 2 ) = T( q_in.z() );
        }
        else
        {
            T sin_mod_delta = std::sin( mod_delta ) / mod_delta;
            res( 0 ) = q_in.x() / sin_mod_delta;
            res( 1 ) = q_in.y() / sin_mod_delta;
            res( 2 ) = q_in.z() / sin_mod_delta;
        }
        return res;
    }
    /* #region   */

    mat_66 refine_cov( int idx = 0 )
    {
        Eigen::Quaterniond q_last, q_icp;
        vec_3              t_last, t_icp;
        double             q_icp_d[ 4 ], r_last_d[ 3 ];
        Jet_6              jet_q_last[ 4 ], jet_q_icp[ 4 ], jet_q_res[ 4 ];
        Jet_6              jet_r_last[ 3 ], jet_r_icp[ 3 ], jet_r_res[ 3 ];
        Jet_6              jet_t_last[ 3 ], jet_t_icp[ 3 ], jet_t_res[ 3 ];
        q_last = m_trajectory_R[ idx - 1 ];
        t_last = m_trajectory_T[ idx - 1 ];
        q_icp = q_last.inverse() * m_trajectory_R[ idx ];
        t_icp = q_last.inverse().toRotationMatrix() * ( m_trajectory_T[ idx ] - m_trajectory_T[ idx - 1 ] );
        q_icp_d[ 0 ] = q_icp.w();
        q_icp_d[ 1 ] = q_icp.x();
        q_icp_d[ 2 ] = q_icp.y();
        q_icp_d[ 3 ] = q_icp.z();

        //ceres::QuaternionToAngleAxis(q_icp_d, r_last_d);
        vec_3 vec_temp = ceres_quadternion_parameterization( q_icp );
        //cout <<"To x delta: " << vec_temp.transpose() << " | " <<q_icp.coeffs().transpose() << endl;
        for ( size_t i = 0; i < 3; i++ )
        {
            //jet_r_icp[i].a = r_last_d[i];
            jet_r_icp[ i ].a = vec_temp( i );
            jet_t_icp[ i ].a = t_icp( i );

            jet_r_icp[ i ].v[ i ] = 1.0;
            jet_t_icp[ i ].v[ i + 3 ] = 1.0;
        }
        // ceres::AngleAxisToQuaternion(jet_r_icp, jet_q_icp );
        jet_q_last[ 0 ] = Jet_6( q_last.w() );
        jet_q_last[ 1 ] = Jet_6( q_last.x() );
        jet_q_last[ 2 ] = Jet_6( q_last.y() );
        jet_q_last[ 3 ] = Jet_6( q_last.z() );
        ceres_quadternion_delta( jet_q_last, jet_r_icp, jet_q_res );

        // ceres::QuaternionProduct(jet_q_last, jet_q_icp, jet_q_res);
        ceres::QuaternionToAngleAxis( jet_q_res, jet_r_res );
        ceres::QuaternionRotatePoint( jet_q_last, jet_t_icp, jet_t_res );
        jet_t_res[ 0 ] += Jet_6( t_last( 0 ) );
        jet_t_res[ 1 ] += Jet_6( t_last( 1 ) );
        jet_t_res[ 2 ] += Jet_6( t_last( 2 ) );

        mat_66 res_cov;
        mat_66 jacobian_mat;
        jacobian_mat.setZero();
        for ( int i = 0; i < 6; i++ )
        {
            for ( int j = 0; j < 3; j++ )
            {
                jacobian_mat( j, i ) = jet_r_res[ j ].v[ i ];
                jacobian_mat( j + 3, i ) = jet_t_res[ j ].v[ i ];
            }
        }
        //cout << q_last.toRotationMatrix() << endl;
        //cout <<std::setprecision(2) <<jacobian_mat << endl;
        res_cov = m_trajectory_cov_mat[ idx ].inverse();
        res_cov = jacobian_mat * res_cov * jacobian_mat.transpose();
        res_cov = res_cov;
        if ( 0 )
        {
            res_cov = mat_66::Identity() * 0.01;
        }
        //cout << "----- " << endl;
        //cout <<std::setprecision(2) << res_cov << endl;
        return res_cov;
    }
};

//ANCHOR global_ekf
class Global_ekf
{
  public:
    typedef Eigen::Matrix<double, 18, 18, Eigen::DontAlign> mat_18_18;
    typedef Eigen::Matrix<double, 42, 42, Eigen::DontAlign> mat_42_42;
    typedef Eigen::Matrix<double, 18, 1, Eigen::DontAlign>  vec_18;
    typedef Eigen::Matrix<double, 42, 1, Eigen::DontAlign>  vec_42;
    
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> m_lidar_initial_extrinsic_R_vec;
    std::vector<vec_3, Eigen::aligned_allocator<vec_3>>                           m_lidar_initial_extrinsic_T_vec;

    typedef ceres::Jet<double, 18> Jet_18;
    typedef ceres::Jet<double, 30> Jet_30;
    typedef ceres::Jet<double, 42> Jet_42;
    Eigen::Matrix<double, 42, 1, Eigen::DontAlign>  m_full_state;
    Eigen::Matrix<double, 42, 42, Eigen::DontAlign> m_full_sigma;
    Eigen::Matrix<double, 42, 1, Eigen::DontAlign>  m_x_g_kplus1[ 5 ];
    mat_42_42                      m_sigma_g_kplus1[ 5 ];
    double                         m_current_time = 0;
    std::vector<std::string>       m_file_name_vec;
    std::vector<Lidar_agent *>     m_lidar_agent_vec;
    std::vector<Local_lidar_ekf *> m_local_lidar_ekf_io_vec;
    std::vector<int>               m_play_index_vec;
    //ofstream* m_ofs = nullptr;
    FILE *      m_fp = nullptr;
    std::string m_log_name;

    int                         m_lidars_count = 0;
    std::shared_ptr<std::mutex> m_log_mutex;
    std::shared_ptr<std::mutex> m_val_mutex;
    int                         m_if_dump_all_traj = 1;
    Global_ekf()
    {
        m_log_mutex = std::make_shared<std::mutex>();
        m_val_mutex = std::make_shared<std::mutex>();
        m_log_name = std::string(TEMP_LOG_SAVE_DIR).append( "./ekf_0.log" );
    };

    ~Global_ekf()
    {
        if ( m_fp != nullptr )
        {
            fflush( m_fp );
            fclose( m_fp );
        }
    };

    void set_file_name_vector( std::vector<std::string> file_name_vec )
    {
        m_file_name_vec = file_name_vec;
        m_lidars_count = m_file_name_vec.size();
    };

    void init_log( std::string log_name = std::string(TEMP_LOG_SAVE_DIR).append("./ekf_0.log" ) )
    {
        m_log_name = log_name;
    }

    void dump_to_log( int idx )
    {
        std::unique_lock<std::mutex> lock( *m_log_mutex );
        double                       q_filtered[ 4 ];

        vec_12 save_vec;
        if ( m_fp == nullptr )
        {
            m_fp = fopen( m_log_name.c_str(), "w+" );
        }
        if ( m_fp != nullptr )
        {
            fprintf( m_fp, "%f ", m_current_time );

            save_vec = m_full_state.block( 0, 0, 12, 1 );
            dump_vector( m_fp, save_vec );

            save_vec = m_full_state.block( 0, 0, 12, 1 );
            dump_vector( m_fp, save_vec );

            Eigen::Matrix<double, 30, 1> extrinsic_vec = m_full_state.block( 12, 0, 30, 1 );
            dump_vector( m_fp, extrinsic_vec );

            fprintf( m_fp, "\r\n" );
            fflush( m_fp );
        }
    }
    void reset_extrinsic(int id = -1);
    void prediction( int lidar_idx, int frame_idx, double new_time );

    template <typename T>
    void printf_ceres_vec( T vec, int size, std::string name = string( "" ) )
    {
        cout << name << " = ";
        cout << std::setprecision( 3 );

        for ( int i = 0; i < size; i++ )
        {
            cout << vec( i ).a << " ";
        }
        cout << endl;
    }

    template <typename T>
    void printf_ceres_array( T vec, int size, std::string name = string( "" ) )
    {
        cout << name << " = ";
        cout << std::setprecision( 5 );

        for ( int i = 0; i < size; i++ )
        {
            cout << vec[ i ].a << " ";
        }
        cout << endl;
    }

    template <typename T>
    T check_safe_update( T _temp_vec )
    {
        // cout<< std::setprecision(2) << "Delta vec = " << _temp_vec.transpose() << endl;

        T temp_vec = _temp_vec;
        if ( std::isnan( temp_vec( 0, 0 ) ) )
        {
            temp_vec.setZero();
            return temp_vec;
        }
        // return temp_vec;
        // return temp_vec;
        double angular_dis = temp_vec.block( 0, 0, 3, 1 ).norm() * 57.3;
        double pos_dis = temp_vec.block( 3, 0, 3, 1 ).norm();
        if ( angular_dis >= 20 || pos_dis > 1 )
        {
            printf( "Angular dis = %.2f, pos dis = %.2f\r\n", angular_dis, pos_dis );
            temp_vec.setZero();
        }
        return temp_vec;
    }

    template <typename T>
    T quaternion_ceres2eigen( T *ceres_q, Eigen::Quaternion<T> &eigen_q )
    {
        eigen_q.w() = ceres_q[ 0 ];
        eigen_q.x() = ceres_q[ 1 ];
        eigen_q.y() = ceres_q[ 2 ];
        eigen_q.z() = ceres_q[ 3 ];
    }

    template <typename T>
    T quaternion_eigen2ceres( Eigen::Quaternion<T> &eigen_q, T *ceres_q )
    {
        ceres_q[ 0 ] = eigen_q.w();
        ceres_q[ 1 ] = eigen_q.x();
        ceres_q[ 2 ] = eigen_q.y();
        ceres_q[ 3 ] = eigen_q.z();
    }

    vec_6 find_e( vec_6 T_g, vec_6 T_i )
    {
        vec_6  res_e;
        double q_g_d[ 4 ], q_i_d[ 4 ], q_temp_d[ 4 ];
        vec_3  t_g, t_i, t_temp;

        Eigen::Quaterniond q_g, q_i, q_res;
        ceres::AngleAxisToQuaternion( T_g.data(), q_g_d );
        ceres::AngleAxisToQuaternion( T_i.data(), q_i_d );
        quaternion_ceres2eigen( q_g_d, q_g );
        quaternion_ceres2eigen( q_i_d, q_i );
        t_g = T_g.block( 3, 0, 3, 1 );
        t_i = T_i.block( 3, 0, 3, 1 );
        q_res = q_i * q_g.inverse();
        t_temp = -( q_res ).toRotationMatrix() * t_g + t_i;
        quaternion_eigen2ceres( q_res, q_temp_d );
        ceres::QuaternionToAngleAxis( q_temp_d, res_e.data() );
        res_e.block( 3, 0, 3, 1 ) = t_temp;
        return res_e;
    }

    template <typename T>
    Eigen::Matrix<T, 6, 1> poses_product( Eigen::Matrix<T, 6, 1> T_g, Eigen::Matrix<T, 6, 1> T_i ) //T_res = T_i * T_g
    {
        Eigen::Matrix<T, 6, 1> res_e;
        T q_g_d[ 4 ], q_i_d[ 4 ], q_temp_d[ 4 ];
        Eigen::Matrix<T, 3, 1> t_g, t_i, t_temp;

        Eigen::Quaternion<T> q_g, q_i, q_res;
        ceres::AngleAxisToQuaternion( T_g.data(), q_g_d );
        ceres::AngleAxisToQuaternion( T_i.data(), q_i_d );
        quaternion_ceres2eigen( q_g_d, q_g );
        quaternion_ceres2eigen( q_i_d, q_i );
        t_g = T_g.block( 3, 0, 3, 1 );
        t_i = T_i.block( 3, 0, 3, 1 );
        q_res = q_i * q_g;
        t_temp = q_i * t_g + t_i;
        quaternion_eigen2ceres( q_res, q_temp_d );
        ceres::QuaternionToAngleAxis( q_temp_d, res_e.data() );
        res_e.block( 3, 0, 3, 1 ) = t_temp;
        return res_e;
    }

    template <typename T>
    void poses_product_array( T *T_g, T *T_i, T *T_res )
    {
        Eigen::Matrix<T, 6, 1> eigen_t_g, eigen_t_l, eigen_t_res;
        for ( size_t i = 0; i < 6; i++ )
        {
            eigen_t_g( i ) = T_g[ i ];
            eigen_t_l( i ) = T_i[ i ];
        }
        eigen_t_res = poses_product<T>( eigen_t_g, eigen_t_l );
        for ( size_t i = 0; i < 6; i++ )
        {
            T_res[ i ] = eigen_t_res( i );
        }
    }

    template <typename T>
    Eigen::Matrix<T, 6, 1> pose_inverse( Eigen::Matrix<T, 6, 1> T_g )
    {
        Eigen::Matrix<T, 6, 1> res_e;
        T q_g_d[ 4 ], q_i_d[ 4 ], q_temp_d[ 4 ];
        Eigen::Matrix<T, 3, 1> t_g, t_i, t_temp;
        Eigen::Quaternion<T> q_g, q_i, q_res;

        ceres::AngleAxisToQuaternion( T_g.data(), q_g_d );
        quaternion_ceres2eigen( q_g_d, q_g );
        t_g = T_g.block( 3, 0, 3, 1 );
        q_res = q_g.inverse();
        t_temp = q_res.toRotationMatrix() * ( t_g ) *T( -1.0 );
        quaternion_eigen2ceres( q_res, q_temp_d );
        ceres::QuaternionToAngleAxis( q_temp_d, res_e.data() );
        res_e.block( 3, 0, 3, 1 ) = t_temp;
        return res_e;
    }

    template <typename T>
    void pose_inverse( T *T_g, T *T_res )
    {
        Eigen::Matrix<T, 6, 1> eigen_t_g, eigen_t_res;
        for ( size_t i = 0; i < 6; i++ )
        {
            eigen_t_g( i ) = T_g[ i ];
        }
        eigen_t_res = pose_inverse<T>( eigen_t_g );
        for ( size_t i = 0; i < 6; i++ )
        {
            T_res[ i ] = eigen_t_res( i );
        }
    }

    template <typename T>
    void state_to_eigen_pose( T pose_prediction, Eigen::Quaterniond &q_prediction, vec_3 &t_prediction )
    {
        double q_temp[ 4 ];
        ceres::AngleAxisToQuaternion( pose_prediction.data(), q_temp );
        quaternion_ceres2eigen( q_temp, q_prediction );
        t_prediction = pose_prediction.block( 3, 0, 3, 1 );
    }

    void get_prediction_of_idx_lidar( int lidar_idx, Eigen::Quaterniond &q_prediction, vec_3 &t_prediction )
    {
        std::unique_lock<std::mutex> lock( *m_val_mutex );
        vec_42                       full_vector;
        full_vector = m_x_g_kplus1[lidar_idx];
        vec_6 pose_prediction;
        poses_product_array( full_vector.data(), &( full_vector.data()[ 12 + 6 * lidar_idx ] ), pose_prediction.data() );
        state_to_eigen_pose( pose_prediction, q_prediction, t_prediction );
    }

    void ekf_measureament_update( int lidar_idx, int frame_idx, double new_time, double obs_gain_para = 1.0, int if_fix_rt = 0 );
    void init();

};
