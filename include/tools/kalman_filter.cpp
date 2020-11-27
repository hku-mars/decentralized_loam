#include "lidar_agent.hpp"

Kalman_filter::~Kalman_filter(){};

Kalman_filter::Kalman_filter()
{
    m_if_verbose_screen_printf = 0;
    // cout  << "Initialization of kalman filter" << std::endl;
    init();
};

void Kalman_filter::init()
{
    m_kl_X_hat.setZero();
    m_kl_X_minus.setZero();
    m_kl_A.setIdentity();
    m_kl_Q.setIdentity();
    m_kl_H.setIdentity();
    m_kl_P.setIdentity();
    m_kl_P_minus.setIdentity();
    m_kl_Q.setIdentity();
    m_kl_R.setIdentity();
    m_kl_Identity.setIdentity();

    set_Q( nullptr, 0.1 );
    set_R( nullptr, 10 );
}

void Kalman_filter::set_state( const vec_6 &mat_x )
{
    // state: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    m_kl_X_hat = mat_x;
}

void Kalman_filter::set_Q( const mat_66 *mat_Q, double s )
{
    if ( mat_Q != nullptr )
    {
        m_kl_Q = ( *mat_Q ) * s;
    }
    else
    {
        m_kl_Q *= s;
    }
}

void Kalman_filter::set_R( const mat_66 *mat_R, double s )
{
    if ( mat_R != nullptr )
    {
        m_kl_R = ( *mat_R ) * s;
    }
    else
    {
        m_kl_R *= s;
    }
}

void Kalman_filter::set_A( const mat_66 &mat_A )
{
    m_kl_A = mat_A;
}

void Kalman_filter::set_delta_t( double dt_s )
{
    // A is 6X6
    // 1, 0, 0, dt, 0, 0
    // 0, 1, 0, 0, dt, 0
    // 0, 0, 1, 0, 0, dt
    // 0, 0, 0, 1, 0, 0
    // 0, 0, 0, 0, 1, 0
    // 0, 0, 0, 0, 0, 1
    mat_66 mat_A;
    mat_A.setIdentity();
    mat_A( 0, 3 ) = dt_s;
    mat_A( 1, 4 ) = dt_s;
    mat_A( 2, 5 ) = dt_s;
    // screen_out <<mat_A <<endl;
    set_A( mat_A );
}

void Kalman_filter::prediction()
{
    if ( m_if_has_prediction )
    {
        return;
    }
    m_kl_X_minus = m_kl_A * m_kl_X_hat;
    m_kl_P_minus = m_kl_A * m_kl_P * ( m_kl_A.transpose() ) + m_kl_Q;
    m_if_has_prediction = 1;
    return;
}

void Kalman_filter::measurement_update( const vec_6 &mat_Z )
{
    m_kl_K = m_kl_P_minus * ( m_kl_H.transpose() ) * ( ( m_kl_H * m_kl_P_minus * ( m_kl_H.transpose() ) + m_kl_R ).inverse() );
    m_kl_X_hat = m_kl_X_minus + m_kl_K * ( mat_Z - m_kl_H * m_kl_X_minus );
    m_kl_P = ( m_kl_Identity - m_kl_K * m_kl_H ) * m_kl_P_minus;
    m_if_has_prediction = 0;
}
