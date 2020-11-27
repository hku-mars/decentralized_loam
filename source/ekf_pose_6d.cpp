#include "ekf_pose_6d.hpp"
#define IF_INDEPENDENT_MODE 0
#define EKF_OBS_GAIN 10.0

void Global_ekf::init()
{
    m_full_state.setZero();
    init_log();
    reset_extrinsic() ;
    // m_full_sigma.setOnes()*100.0;

    // m_full_sigma.block( 12, 12, 6, 6 ) *= 0.0;

    // cout << "m_full_sigma:\r\n"
    //      << m_full_sigma << endl;
    m_lidar_agent_vec.resize( m_lidars_count );
    m_play_index_vec.resize( m_lidars_count );
#pragma omp parallel for
    for ( int i = 0; i < m_lidars_count; i++ )
    {
        m_lidar_agent_vec[ i ] = new Lidar_agent();
        m_lidar_agent_vec[ i ]->load_from_file( m_file_name_vec[ i ] );
        cout << "Load from " << m_file_name_vec[ i ] << endl;
        cout << "Start timestamp = " << m_lidar_agent_vec[ i ]->m_trajectory_time.front() << endl;
    }

    for ( size_t i = 0; i < m_lidars_count; i++ )
    {
        m_local_lidar_ekf_io_vec.push_back( new Local_lidar_ekf( *m_lidar_agent_vec[ i ] ) );
        m_local_lidar_ekf_io_vec.back()->reset();
        m_play_index_vec[ i ] = 0;
    }
}

void Global_ekf::reset_extrinsic(int id)
{
    cout << "Call reset extrinsic" << endl;
    std::unique_lock<std::mutex> lock(*m_val_mutex);

    for ( int i = 0; i < 5; i++ )
    {
        vec_3 angle_axis;
        m_full_state.block( 12 + 6 * i, 0, 3, 1 ) = vec_3(0,0,0);
        m_full_state.block( 15 + 6 * i, 0, 3, 1 ) = m_lidar_initial_extrinsic_T_vec[ i ];
    }

    if ( id < 0 )
    {
        m_full_state.block( 12, 0, 30, 1 ).setZero();
        m_full_sigma.setIdentity();
        // m_full_sigma.block(12,12,30,30) /= 100000.0;
        for ( int i = 0; i < 5; i++ )
        {
            m_full_sigma.block( 12 + i * 6, 12 + i * 6, 3, 3 ) = mat_33::Identity() * 1.0;
            m_full_sigma.block( 15 + i * 6, 15 + i * 6, 3, 3 ) = mat_33::Identity() * 0.001;
        }
    }
}

void Global_ekf::prediction( int lidar_idx, int frame_idx, double new_time )
{
    std::unique_lock<std::mutex> lock(*m_val_mutex);
    double delta_time = new_time - m_current_time;
    // cout << "delta_time = " << delta_time << endl;
    Jet_42    x_g_k[ 42 ], x_g_kplus1[ 42 ];
    Jet_42    q_g_k[ 4 ], q_g_delta[ 4 ], q_g_kplus1[ 4 ];
    Jet_42    delta_t[ 3 ], delta_r[ 3 ], t_kplus1[ 3 ], bar_r_kplus1[ 3 ];
    mat_42_42 sigma_g;
    mat_42_42 mat_Gg;
    for ( int i = 0; i < 42; i++ )
    {
        x_g_k[ i ].a = m_full_state( i );
    }

    for ( int i = 0; i < 42; i++ )
    {
        x_g_k[ i ].v[ i ] = 1.0;
        x_g_kplus1[ i ] = x_g_k[ i ];
    }

    //if ( lidar_idx == 0 )
    if ( IF_INDEPENDENT_MODE )
    {
        for ( int i = 12; i < 42; i++ )
        {
            x_g_k[ i ].v[ i ] = 0.0;
            x_g_kplus1[ i ].v[ i ] = 0.0;
        }
    }

    sigma_g = m_full_sigma;

    for ( size_t i = 0; i < 3; i++ )
    {
        delta_r[ i ] = x_g_k[ 6 + i ] * Jet_42( delta_time );
        delta_t[ i ] = x_g_k[ 9 + i ] * Jet_42( delta_time );
    }

    ceres::AngleAxisToQuaternion( x_g_k, q_g_k );
    ceres::AngleAxisToQuaternion( delta_r, q_g_delta );
    ceres::QuaternionProduct( q_g_k, q_g_delta, q_g_kplus1 );
    ceres::QuaternionToAngleAxis( q_g_kplus1, x_g_kplus1 );

    for ( int i = 0; i < 3; i++ )
    {
        x_g_kplus1[ 3 + i ] = delta_t[ i ] + x_g_k[ 3 + i ];
    }

    for ( int i = 0; i < 42; i++ )
    {
        m_x_g_kplus1[ lidar_idx ][ i ] = x_g_kplus1[ i ].a;
    }

    for ( int i = 0; i < 42; i++ )
    {
        for ( int j = 0; j < 42; j++ )
        {
            mat_Gg( i, j ) = x_g_kplus1[ i ].v[ j ];
        }
    }

        m_sigma_g_kplus1[ lidar_idx ] = ( mat_Gg * sigma_g * mat_Gg.transpose() );
        m_sigma_g_kplus1[ lidar_idx ].block( 6, 6, 3, 3 ) += mat_33::Identity() * 0.001; // noise of angular velocity
        m_sigma_g_kplus1[ lidar_idx ].block( 9, 9, 3, 3 ) += mat_33::Identity() * 0.001; // noise of linear velocity
    
    if(IF_INDEPENDENT_MODE)
    {
        m_sigma_g_kplus1[ lidar_idx ].block( 12, 12, 30, 30 ) = sigma_g.block( 12, 12, 30, 30 );
    }

    if ( 0 )
    {
        cout << "--------------------" << endl;
        cout << "X_g_k+1: " << m_x_g_kplus1[ lidar_idx ].transpose() << endl;
        cout << "Sigma_g_k+1: \r\n"
             << m_sigma_g_kplus1[ lidar_idx ] << endl;
    }
}

// template <typename T, typename TT >
// void extend_extrinsic(T & vec_in, TT &vec_res)
void extend_extrinsic( ceres::Jet< double, 42 > *vec_in,
                       Eigen::Matrix< ceres::Jet< double, 42 >, 12, 1, Eigen::DontAlign > &vec_res )
{
    for ( int i = 0; i < 6; i++ )
    {
        vec_res( i + 6 ) = ceres::Jet< double, 42 >( 0.0 );
        for ( int j = 0; j < 1; j++ )
        //int j = 0;
        {
            vec_res( i + 6 ) += vec_in[ i + 12 + 6 * j ] * ceres::Jet< double, 42 >( -1.0 );
        }
    }
}

void Global_ekf::ekf_measureament_update( int lidar_idx, int frame_idx, double new_time , double obs_gain, int if_fix_rt)
{
    if(new_time < m_current_time)
        return;
    if ( m_lidar_agent_vec[ lidar_idx ]->m_trajectory_cov_mat[ frame_idx ].trace() == 0 )
        return;
    std::unique_lock<std::mutex> lock(*m_val_mutex);
    Jet_42 x_g_kplus1[ 42 ];
    Jet_42 q_ei[ 4 ], q_g_kplus1[ 4 ], t_ei_temp[ 3 ];
    Eigen::Matrix< Jet_42, 6, 1, Eigen::DontAlign > jet_z_g, jet_z_o, jet_q_delta;
    Eigen::Matrix< double, 6, 1, Eigen::DontAlign > z_g, z_o, hat_z_o;

    Eigen::Matrix< double, 42, 1, Eigen::DontAlign > bar_z_g, star_z_g;

    Eigen::Matrix< double, 6, 42, Eigen::DontAlign >  mat_Zg;
    Eigen::Matrix< double, 6, 6, Eigen::DontAlign >   sigma_zg;
    Eigen::Matrix< double, 42, 42, Eigen::DontAlign > sigma_g_kplus1, star_sigma_g;

    Eigen::Matrix< double, 42, 6, Eigen::DontAlign > mat_Kg;

    Eigen::Matrix< double, 6, 42, Eigen::DontAlign > mat_Hg;

    sigma_g_kplus1.setZero();
    sigma_g_kplus1 = m_sigma_g_kplus1[ lidar_idx ]*1.0;
    sigma_g_kplus1.block(0, 0, 12 , 12) = m_sigma_g_kplus1[ lidar_idx ].block(0, 0, 12 , 12);
    // for(int i =0 ; i< 5; i++)
    if(IF_INDEPENDENT_MODE)
    {
        int i = lidar_idx;
        sigma_g_kplus1.block(12+6*i, 12+6*i, 6 , 6) *= 0.995;
    }
    //sigma_g_kplus1.block( 12, 12, 6, 6m_sigma_g_kplus1 ) = m_local_lidar_ekf_io_vec[ lidar_idx ].m_covariance_extrinsic * 0.9995;

    for ( int i = 0; i < 42; i++ )
    {
        x_g_kplus1[ i ].a = m_x_g_kplus1[ lidar_idx ]( i );
        x_g_kplus1[ i ].v[ i ] = 1.0;
    }

    for ( int i = 0; i < 42; i++ )
    {
        bar_z_g[ i ] = x_g_kplus1[ i ].a;
    }

    if ( lidar_idx == 0 )
    {
        for ( int i = 12; i < 18; i++ )
        {
            x_g_kplus1[ i ].v[ i ] = 0.0;
        }
    }

    if(if_fix_rt)
    {
        for ( int i = 12; i < 42; i++ )
        {
            x_g_kplus1[ i ].v[ i ] = 0.0;
        }
    }

    poses_product_array( x_g_kplus1, &x_g_kplus1[ 12 + 6 * lidar_idx ], jet_z_g.data() );
    if ( 0 ) // for testing only
    {
        Eigen::Quaterniond q_pre;
        vec_3              t_pre;
        get_prediction_of_idx_lidar( lidar_idx, q_pre, t_pre );
        printf_ceres_vec( jet_z_g, 6, "jet_z_g" );
        cout << "i-th lidar prediction: " << q_pre.coeffs().transpose() << " | " << t_pre.transpose() << endl;
    }
    for ( int i = 0; i < 42; i++ )
    {
        for ( int j = 0; j < 6; j++ )
        {
            mat_Zg( j, i ) = jet_z_g( j ).v[ i ];
        }
    }

    Eigen::Matrix< Jet_30, 6, 6, Eigen::DontAlign > jet_sig_zg_inv, jet_sig_zi_inv, jet_sig_zo;

    sigma_zg = mat_Zg * sigma_g_kplus1 * mat_Zg.transpose() + mat_66::Identity() * 0;

    Eigen::Quaterniond hat_q_z = m_lidar_agent_vec[ lidar_idx ]->m_trajectory_R[ frame_idx ];
    vec_3              hat_t_z = m_lidar_agent_vec[ lidar_idx ]->m_trajectory_T[ frame_idx ];
    double             hat_q_double[ 4 ];
    hat_q_double[ 0 ] = hat_q_z.w();
    hat_q_double[ 1 ] = hat_q_z.x();
    hat_q_double[ 2 ] = hat_q_z.y();
    hat_q_double[ 3 ] = hat_q_z.z();
    ceres::QuaternionToAngleAxis( hat_q_double, hat_z_o.data() );
    hat_z_o.block( 3, 0, 3, 1 ) = hat_t_z;
    mat_66 hat_cov, icp_cov;

    jet_z_o = hat_z_o.template cast< Jet_42 >();

    if ( if_mode_delta )
    {
        jet_q_delta = poses_product< Jet_42 >( jet_z_o, pose_inverse< Jet_42 >( jet_z_g ) );
        icp_cov = m_lidar_agent_vec[ lidar_idx ]->m_trajectory_cov_mat[ frame_idx ].inverse() * EKF_OBS_GAIN + mat_66::Identity() * 0.001;
    }
    else
    {
        hat_cov = m_local_lidar_ekf_io_vec[ lidar_idx ]->refine_cov( frame_idx )*EKF_OBS_GAIN;
    }
    vec_6 q_delta_double;

    for ( int i = 0; i < 6; i++ )
    {
        z_g( i ) = jet_z_g( i ).a;
    }

    for ( int j = 0; j < 6; j++ )
    {
        q_delta_double( j ) = jet_q_delta( j ).a;
        for ( int i = 0; i < 42; i++ )
        {
            if ( if_mode_delta )
            {
                mat_Hg( j, i ) = jet_q_delta[ j ].v[ i ];
            }
            else
            {
                mat_Hg( j, i ) = jet_z_g[ j ].v[ i ];
            }
        }
    }

    if ( m_if_dump_all_traj )
    {
        cout << std::setprecision( 3 ) << "hat_z_o       : " << hat_z_o.transpose() << endl;
        cout << std::setprecision( 3 ) << "q_delta_double: " << q_delta_double.transpose() << endl;
        cout << std::setprecision( 3 ) << "q_bar         : " << poses_product< double >( q_delta_double, z_g ).transpose() << endl;
        printf_ceres_vec( jet_z_g, 6, "jet_z_g" );
        printf_ceres_array( x_g_kplus1, 12, "center_state" );
        //printf_ceres_vec( jet_z_o, 6, "jet_z_o" );
    }

    if ( 1 )
    {
        if ( if_mode_delta )
        {
            // cout << setprecision(2) << "mat_hg = \r\n" << mat_Hg << endl;
            // cout << std::setprecision( 2 ) << "icp_cov:\r\n" << icp_cov << endl;
            if ( std::isnan( ( icp_cov( 0, 0 ) ) ) )
            {
                cout << std::setprecision( 2 ) << "icp_cov:\r\n"
                     << icp_cov << endl;
                return;
            }
            mat_Kg = sigma_g_kplus1 * ( mat_Hg.transpose() ) * ( ( mat_Hg * sigma_g_kplus1 * ( mat_Hg.transpose() ) + icp_cov ).inverse() );
        }
        else
        {
            mat_Kg = sigma_g_kplus1 * ( mat_Hg.transpose() ) * ( ( mat_Hg * sigma_g_kplus1 * ( mat_Hg.transpose() ) + hat_cov ).inverse() );
        }
    }
    else
    {
        mat_Kg = sigma_g_kplus1 * ( mat_Hg.transpose() ) * ( ( mat_Hg * sigma_g_kplus1 * ( mat_Hg.transpose() ) + sigma_zg + hat_cov ).inverse() );
    }

    vec_6 hat_minus_z_g = hat_z_o - z_g;

    vec_42 temp_vec_42;
    if ( if_mode_delta )
    {
        temp_vec_42 = check_safe_update< vec_42 >( mat_Kg * ( -q_delta_double ) );
    }
    else
    {
        vec_6 hat_minus_z_g = hat_z_o - z_g;
        temp_vec_42 = check_safe_update< vec_42 >( mat_Kg * hat_minus_z_g );
    }
    if ( std::isnan( temp_vec_42( 0 ) ) )
        return;
    star_z_g = bar_z_g + temp_vec_42;

    if ( 0 )
    { //cout << std::setprecision( 2 ) << "mat_Hg:\r\n" << mat_Hg << endl;
        //cout << std::setprecision( 2 ) << "mat_Kg:\r\n" << mat_Kg << endl;
        // cout << std::setprecision( 2 ) << "temp_vec_18:\r\n" << temp_vec_18.transpose() << endl;

        // star_z_i.block(0,0,6,1) = apple_g2l(star_z_g.block(0,0,6,1), T_e );
        printf_ceres_vec( jet_z_g, 6, "jet_z_g" );
        cout << std::setprecision( 3 ) << new_time << " |raw_hat_z = " << hat_q_z.coeffs().transpose() << " -- " << hat_t_z.transpose() << endl;
        // cout <<std::setprecision(3) << new_time<< " | " << star_z_o.transpose() << endl;
        // cout << std::setprecision( 3 ) << new_time << " |star_z_g = " << star_z_g.transpose() << endl;
        cout << std::setprecision( 3 ) << new_time << " |star_z_g = " << star_z_g.block( 0, 0, 12, 1 ).transpose() << endl;
    }

    star_sigma_g = ( Eigen::Matrix< double, 42, 42, Eigen::DontAlign >::Identity() - mat_Kg * mat_Hg ) * sigma_g_kplus1;

    if ( m_if_dump_all_traj )
    {
        m_local_lidar_ekf_io_vec[ lidar_idx ]->m_state_X_curr = star_z_g.block( 0, 0, 12, 1 );
        //m_local_lidar_ekf_io_vec[ lidar_idx ].m_state_X_curr.block(0,0,6,1) = poses_product(m_local_lidar_ekf_io_vec[ lidar_idx ].m_state_X_curr.block(0,0,6,1), star_z_g.block(12+6*lidar_idx, 0, 6,1 );
        m_local_lidar_ekf_io_vec[ lidar_idx ]->m_cov_sigma_curr = star_sigma_g.block( 0, 0, 12, 12 );
        m_local_lidar_ekf_io_vec[ lidar_idx ]->m_state_bar_x_curr.block( 0, 0, 6, 1 ) = poses_product< double >( star_z_g.block( 0, 0, 6, 1 ), star_z_g.block( 12 + 6 * lidar_idx, 0, 6, 1 ) );
    }
    m_full_state = star_z_g;
    m_full_sigma = star_sigma_g;

    // for ( int j = 0; j < 6; j++ )
    // {
    //     m_local_lidar_ekf_io_vec[ j ]->m_state_extrinsic = star_z_g.block( 12+6*j, 0, 6, 1 );
    //     m_local_lidar_ekf_io_vec[ j ]->m_covariance_extrinsic = star_sigma_g.block( 12+6*j, 12+6*j, 6, 6 );
    // }

    m_current_time = new_time;
}