#pragma once
#include <stdio.h>
#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Eigen>
namespace Common_tools
{

    /****** Usage *****/
    // ceres::Problem::EvaluateOptions eval_options;
    // std::vector<double> gradients;
    // ceres::CRSMatrix jacobian_matrix;
    // problem.Evaluate( eval_options, &total_cost, &residuals, &gradients, &jacobian_matrix );
    // save_ceres_crs_matrix_to_txt("/home/ziv/jacobian.txt", jacobian_matrix);

// Refer to crs_matrix.h @  https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/crs_matrix.h

inline Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > crs_to_eigen_matrix(const ceres::CRSMatrix &crs_matrix )
{
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > dy_mat;
    dy_mat.resize( crs_matrix.num_rows, crs_matrix.num_cols );
    dy_mat.setZero();
    int val_idx = 0;
    for ( size_t i = 0; i < crs_matrix.rows.size() - 1; i++ )
    {
        int curr_row = i;
        for ( int ii = crs_matrix.rows[ i ]; ii < crs_matrix.rows[ i + 1 ]; ii++ )
        {
            int curr_col = crs_matrix.cols[ ii ];
            //printf( "[%d. %d][%d. %d]--%lf\r\n ", crs_matrix.num_rows, crs_matrix.num_cols, curr_row, curr_col, crs_matrix.values[ val_idx ] );
            dy_mat( curr_row, curr_col ) = crs_matrix.values[ val_idx ];
            val_idx++;
        }
    }
    //printf( "Matrix detisity = %.2f \r\n", ( float ) val_idx / ( float ) ( crs_matrix.num_rows * crs_matrix.num_cols ) );
    return dy_mat;
}

template<typename T>
Eigen::Matrix<T, 3, 1> ceres_quadternion_parameterization(const Eigen::Quaternion<T> &q_in)
{
    Eigen::Matrix< T, 3, 1 > res;
    T                        mod_delta = acos( q_in.w() );
    if ( abs(q_in.w()-1) <= 1e-10 )
    {
        res( 0 ) = 0;
        res( 1 ) = 0;
        res( 2 ) = 0;
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

template<typename T, typename TT>
void ceres_quadternion_delta( T * t_curr, TT &t_delta )
{
    // Refer from: http://ceres-solver.org/nnls_modeling.html#_CPPv2N5ceres26QuaternionParameterizationE
    // cout << "Input delta:" << t_delta.transpose() << endl;
    T mod_delta = t_delta.norm();
    if(mod_delta == 0) //equal to identity.
    {
        return;
    }
    T sin_by_mod_delta = std::sin(mod_delta )/ mod_delta;

    Eigen::Quaternion<T> q_delta;
    Eigen::Map< Eigen::Quaternion<T> >                  q_w_curr( t_curr );

    q_delta.w() = std::cos(mod_delta);
    q_delta.x() = sin_by_mod_delta * t_delta(0);
    q_delta.y() = sin_by_mod_delta * t_delta(1);
    q_delta.z() = sin_by_mod_delta * t_delta(2);
    // cout << q_delta.coeffs().transpose() << endl;
    q_w_curr = q_delta*(q_w_curr);
}

template <typename T>
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > std_vector_to_eigen_matrix(std::vector<T> & in_vector, int number_or_cols = 0)
{
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > res_mat;
    size_t total_size = in_vector.size();
    size_t num_cols = 1;
    if(number_or_cols)
    {
        num_cols = number_or_cols;
        assert((total_size % num_cols) ==0 );
    }
    res_mat.resize(  total_size/num_cols, num_cols );
    for(size_t i = 0; i< total_size; i++)
    {
        res_mat(i) = in_vector[i];
    }
    //cout << "Size of res_mat: " << res_mat.rows() << " X " << res_mat.cols() << endl;
    return res_mat;
}


inline void save_ceres_crs_matrix_to_txt( std::string file_name, ceres::CRSMatrix &crs_matrix, int do_print=  0 )
{
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > dy_mat = crs_to_eigen_matrix(crs_matrix);
    FILE *fp;
    fp = fopen( file_name.c_str(), "w+" );
    if ( do_print )
    {
        for ( int j = 0; j < dy_mat.rows(); j++ )
        {
            for ( int i = 0; i < dy_mat.cols(); i++ )
            {
                fprintf( fp, "%lf ", dy_mat( j, i ) );
            }
        }
    }
    else
    {
        for ( int i = 0; i < dy_mat.cols(); i++ )
        {
            for ( int j = 0; j < dy_mat.rows(); j++ )
            {

                fprintf( fp, "%lf ", dy_mat( j, i ) );
            }
        }
    }

    fclose( fp );
};
    
};