#include "lidar_agent.hpp"
Lidar_agent::Lidar_agent()
{
    m_trajectory_R.reserve( 1e5 );
    m_trajectory_T.reserve( 1e5 );
    m_trajectory_R.push_back( m_q_w_curr );
    m_trajectory_T.push_back( m_t_w_curr );
    m_timer.tic();

    m_state_current.resize( 6, 1 );
    m_state_last.resize( 6, 1 );
    m_mutex = std::make_shared<std::mutex>();
};

Lidar_agent::~Lidar_agent(){};

void Lidar_agent::clear_data()
{
    m_trajectory_time.clear();
    m_trajectory_T.clear();
    m_trajectory_R.clear();
}

void Lidar_agent::init_log( int index )
{
    m_logger.set_log_dir( TEMP_LOG_SAVE_DIR );
    m_logger.init( string( "lidar_" ).append( std::to_string( index ) ).append( ".log" ) );
}

void Lidar_agent::update_history( double current_time, int save_log  )
{
    std::unique_lock<std::mutex> lock( *m_mutex );
    if ( current_time < 0 )
    {
        current_time = m_timer.toc( " ", 0 );
    }
    m_trajectory_time.push_back( current_time );
    m_trajectory_R.push_back( Eigen::Quaterniond( m_q_w_curr ) );
    m_trajectory_T.push_back( Eigen::Matrix<double, 3, 1>( m_t_w_curr ) );
    m_trajectory_cov_mat.push_back( m_cov_mat_curr );
    if ( save_log )
    {
        m_logger.printf( "%f %f %f %f %f %f %f %f", current_time, m_q_w_curr.w(), m_q_w_curr.x(), m_q_w_curr.y(), m_q_w_curr.z(), m_t_w_curr.x(), m_t_w_curr.y(), m_t_w_curr.z() );

        for ( size_t ii = 0; ii < 36; ii++ )
        {
            m_logger.printf( " %f", m_cov_mat_curr( ii ) );
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6, Eigen::DontAlign>> eig( m_cov_mat_curr ); // singular value equal to eigen value
        for ( size_t ii = 0; ii < 6; ii++ )
        {
            m_logger.printf( " %f", eig.eigenvalues()( ii ) );
        }
        m_logger.printf( "\r\n" );
    }
}


void Lidar_agent::update( double current_time, const Eigen::Quaterniond &q_curr, const vec_3 &t_curr, const mat_66 & cov,  int save_log ) 
{
    m_q_w_curr = q_curr;
    m_t_w_curr = t_curr;
    m_cov_mat_curr = cov;
    update_history( current_time, save_log );
}

void Lidar_agent::save_trajectory_to_file( const string &file_name, std::vector<double>                                   trajectory_time,
                                           std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> &trajectory_R,
                                           std::vector<vec_3, Eigen::aligned_allocator<vec_3>> &                          trajectory_T,
                                           std::vector<mat_66, Eigen::aligned_allocator<mat_66>> &                        trajectory_cov )
{
    // cout << "Time size = " << trajectory_time.size() << endl;
    // cout << "Trajectory_R size = " << trajectory_R.size() << endl;
    // cout << "Trajectory_T size = " << trajectory_T.size() << endl;
    assert( trajectory_time.size() == trajectory_R.size() );
    assert( trajectory_time.size() == trajectory_T.size() );
    //assert( trajectory_time.size() == trajectory_cov.size() );

    if(trajectory_cov.size() == 0)
    {
        for (size_t i =0  ;i< m_trajectory_time.size() ; i++)
        {
            trajectory_cov.push_back(mat_66::Identity());
        }

    }
    FILE *fp;
    fp = fopen( file_name.c_str(), "w+" );
    if ( fp != nullptr )
    {
        size_t traj_size = trajectory_time.size();
        for ( size_t i = 0; i < traj_size; i++ )
        {
            fprintf( fp, "%f %f %f %f %f %f %f %f ", trajectory_time[ i ], trajectory_R[ i ].w(), trajectory_R[ i ].x(), trajectory_R[ i ].y(), trajectory_R[ i ].z(),
                     trajectory_T[ i ].x(), trajectory_T[ i ].y(), trajectory_T[ i ].z() );
            for ( size_t ii = 0; ii < 36; ii++ )
            {
                fprintf( fp, "%f ", trajectory_cov[i](ii));
            }
            fprintf( fp, "\r\n");
        }
        fclose( fp );
        screen_out << "save to " << file_name << " successful" << std::endl;
    }
}

void Lidar_agent::save_raw_trajectory_to_file( const string &file_name )
{
    save_trajectory_to_file( file_name, m_trajectory_time, m_trajectory_R, m_trajectory_T, m_trajectory_cov_mat );
}


void Lidar_agent::print_trajectory()
{
    for ( int i = 0; i < m_trajectory_time.size(); i++ )
    {
        double curr_time = m_trajectory_time[ i ];
        auto   q_curr = m_trajectory_R[ i ];
        auto   t_curr = m_trajectory_T[ i ];
        screen_out << curr_time << " | " << q_curr.coeffs().transpose() << " -- " << t_curr.transpose() << std::endl;
    }
}

void Lidar_agent::load_from_file( const string &file_name )
{
    FILE *fp;
    fp = fopen( file_name.c_str(), "r+" );
    double data[ 1000 ] = { 0 };
    if ( fp != nullptr )
    {
        char temp_char[10000];
        cout << "Load trajectory from: " << file_name << endl;
        m_trajectory_time.clear();
        m_trajectory_R.clear();
        m_trajectory_T.clear();
        while ( 1 )
        {
            if ( feof( fp ) )
            {
                break;
            }

            double curr_time;
            int temp_int;
            temp_int = fscanf( fp, "%lf %lf %lf %lf %lf %lf %lf %lf", &curr_time, &m_q_w_curr.w(), &m_q_w_curr.x(), &m_q_w_curr.y(), &m_q_w_curr.z(),
                    &m_t_w_curr.data()[ 0 ], &m_t_w_curr.data()[ 1 ], &m_t_w_curr.data()[ 2 ] );
            for ( size_t i = 0; i < 36; i++ )
            {
                temp_int = fscanf( fp, " %lf", &m_cov_mat_curr.data()[ i ] );
            }

            vec_6 eig_val;
            for ( size_t idx = 0; idx < 6; idx++ )
            {
                temp_int = fscanf( fp, " %lf", &eig_val.data()[ idx ] );
            }
            
            m_trajectory_time.push_back( curr_time );
            m_trajectory_R.push_back( m_q_w_curr );
            m_trajectory_T.push_back( m_t_w_curr );
            m_trajectory_cov_mat.push_back( m_cov_mat_curr );
            m_eig_val_vec.push_back( eig_val );

            fgets (temp_char , 10000 , fp); // change line
        }
        std::cout << "Trajectory have: " << m_trajectory_time.size() << " -- "
                  << m_trajectory_R.size() << " -- "
                  << m_trajectory_T.size() << " poses" << std::endl;
        fclose( fp );
    }
    else
    {
        std::cout << "Open " << file_name << " fail, please check!!!" << endl;
    }
    
}