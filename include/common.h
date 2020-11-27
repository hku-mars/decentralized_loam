// Author: Jiarong Lin          ziv.lin.ljr@gmail.com

#pragma once
// #define TEMP_LOG_SAVE_DIR "/home/ziv/Loam_livox/"
#define TEMP_LOG_SAVE_DIR "./"
#include <cmath>

#include <pcl/point_types.h>
#define printf_line printf( " %s %d \r\n", __FILE__, __LINE__ );
typedef pcl::PointXYZI PointType;
#define MAXIMUM_LIDAR_SIZE 10
#if 1
typedef Eigen::Matrix<double, 3, 1, Eigen::DontAlign> vec_3;
typedef Eigen::Matrix<double, 6, 1, Eigen::DontAlign> vec_6;
typedef Eigen::Matrix<double, 12, 1, Eigen::DontAlign> vec_12;
typedef Eigen::Matrix<double, 3, 3, Eigen::DontAlign> mat_33;
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign> mat_66;
typedef Eigen::Matrix<double, 12, 12, Eigen::DontAlign> mat_12;
typedef Eigen::Matrix<double, 6, 12, Eigen::DontAlign> mat_6_12;
typedef Eigen::Matrix<double, 12, 6, Eigen::DontAlign> mat_12_6;
#else
typedef Eigen::Matrix<double, 6, 1> vec_6;
typedef Eigen::Matrix<double, 3, 1> vec_3;
typedef Eigen::Matrix<double, 3, 3> mat_33;
typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign > mat_66;
// typedef Eigen::Matrix<double, 6, 6 > mat_66;
#endif

typedef Eigen::Matrix<double, 6, 6, Eigen::DontAlign > mat_66_noalign;
inline double rad2deg( double radians )
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}
