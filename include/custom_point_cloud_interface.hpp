#ifndef __CUSTOM_LOAM_LIVOX_MESSAGE_LOAM_LIVOX_CUSTOM_POINT_CLOUD_INTERFACE__
#define __CUSTOM_LOAM_LIVOX_MESSAGE_LOAM_LIVOX_CUSTOM_POINT_CLOUD_INTERFACE__
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <common_tools.h>
#include <dc_loam/Loam_livox_custom_point_cloud.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

using dc_loam::Loam_livox_custom_point_cloud;
struct Custom_point_cloud_interface
{
    std::shared_ptr<dc_loam::Loam_livox_custom_point_cloud> m_msg_ptr;

    static void msg_to_pcl_pc( const dc_loam::Loam_livox_custom_point_cloud &custom_pc_msg , pcl::PointCloud<pcl::PointXYZI > & pcl_pt)
    {
        size_t pt_size = custom_pc_msg.m_point_size;
        assert(custom_pc_msg.m_point_xyz.size() == pt_size*4);
        pcl_pt.points.resize( pt_size );
        for ( size_t i = 0; i < pt_size; i++ )
        {
            pcl_pt.points[ i ].x = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i ];
            pcl_pt.points[ i ].y = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 1 ];
            pcl_pt.points[ i ].z = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 2 ];
            pcl_pt.points[ i ].intensity = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 3 ];
        }
    }

    static void msg_to_pcl_pc( const dc_loam::Loam_livox_custom_point_cloud &custom_pc_msg , pcl::PointCloud<pcl::PointXYZ > & pcl_pt)
    {

        size_t pt_size = custom_pc_msg.m_point_size;
        pcl_pt.points.resize( pt_size );
        assert(custom_pc_msg.m_point_xyz.size() == pt_size*3);
        for ( size_t i = 0; i < pt_size; i++ )
        {
            pcl_pt.points[ i ].x = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i ];
            pcl_pt.points[ i ].y = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 1 ];
            pcl_pt.points[ i ].z = custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 2 ];
        }
    }

    static void msg_from_pcl_pc( dc_loam::Loam_livox_custom_point_cloud &custom_pc_msg, pcl::PointCloud<pcl::PointXYZI > & pcl_pt )
    {
        size_t pt_size = pcl_pt.points.size();
        custom_pc_msg.m_point_size = pt_size;
        custom_pc_msg.m_point_step = 4;
        custom_pc_msg.m_point_xyz.resize( pt_size * custom_pc_msg.m_point_step );
        custom_pc_msg.m_intensity.resize( pt_size );
        custom_pc_msg.m_timestamp.resize( pt_size );
        for ( size_t i = 0; i < pt_size; i++ )
        {
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i ] = pcl_pt.points[ i ].x;
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 1 ] = pcl_pt.points[ i ].y;
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 2 ] = pcl_pt.points[ i ].z;
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 3 ] = pcl_pt.points[ i ].intensity;
        }
    }
    
    static void msg_from_pcl_pc( dc_loam::Loam_livox_custom_point_cloud &custom_pc_msg, pcl::PointCloud<pcl::PointXYZ > & pcl_pt )
    {
        size_t pt_size = pcl_pt.points.size();
        custom_pc_msg.m_point_size = pt_size;
        custom_pc_msg.m_point_step = 3;

        custom_pc_msg.m_point_xyz.resize( pt_size * custom_pc_msg.m_point_step );
        custom_pc_msg.m_intensity.resize( pt_size );
        custom_pc_msg.m_timestamp.resize( pt_size );
        for ( size_t i = 0; i < pt_size; i++ )
        {
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i ] = pcl_pt.points[ i ].x;
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 1 ] = pcl_pt.points[ i ].y;
            custom_pc_msg.m_point_xyz[ custom_pc_msg.m_point_step * i + 2 ] = pcl_pt.points[ i ].z;
        }
    }

    Custom_point_cloud_interface()
    {
    }

    Custom_point_cloud_interface( std::shared_ptr<dc_loam::Loam_livox_custom_point_cloud> msg_ptr )
    {
        set_msg_ptr(msg_ptr);
    };
    
    ~Custom_point_cloud_interface(){};
    
    template <typename T>
    void set_msg_ptr( T msg_ptr)
    {
        m_msg_ptr = std::shared_ptr<dc_loam::Loam_livox_custom_point_cloud>(msg_ptr);
    }
};

inline Loam_livox_custom_point_cloud operator+( const Loam_livox_custom_point_cloud &msg_a, const Loam_livox_custom_point_cloud &msg_b )
{
    Loam_livox_custom_point_cloud msg_res;
    // assert(msg_a.m_point_step == msg_b.m_point_step);
    if ( msg_a.m_point_size == 0 )
    {
        msg_res = msg_b;
        return msg_res;
    }
    msg_res = msg_a;
    msg_res.m_point_step = msg_a.m_point_step;
    msg_res.m_point_size = msg_a.m_point_size + msg_b.m_point_size;
    msg_res.m_point_xyz.insert( msg_res.m_point_xyz.end(), msg_b.m_point_xyz.begin(), msg_b.m_point_xyz.end() );
    msg_res.m_intensity.insert( msg_res.m_intensity.end(), msg_b.m_intensity.begin(), msg_b.m_intensity.end() );
    msg_res.m_timestamp.insert( msg_res.m_timestamp.end(), msg_b.m_timestamp.begin(), msg_b.m_timestamp.end() );
    return msg_res;
}

inline Loam_livox_custom_point_cloud operator+=(Loam_livox_custom_point_cloud& msg_a,const Loam_livox_custom_point_cloud& msg_b )
{   
    if ( msg_a.m_point_size == 0 )
    {
        msg_a = msg_b;
        return msg_a;
    }
    msg_a.m_point_size = msg_a.m_point_size + msg_b.m_point_size;
    msg_a.m_point_xyz.insert( msg_a.m_point_xyz.end() , msg_b.m_point_xyz.begin(), msg_b.m_point_xyz.end());
    msg_a.m_intensity.insert( msg_a.m_intensity.end() , msg_b.m_intensity.begin(), msg_b.m_intensity.end());
    msg_a.m_timestamp.insert( msg_a.m_timestamp.end() , msg_b.m_timestamp.begin(), msg_b.m_timestamp.end());
    return msg_a;
}

struct Data_pair
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Loam_livox_custom_point_cloud m_pc_corner;
    Loam_livox_custom_point_cloud m_pc_full;
    Loam_livox_custom_point_cloud m_pc_plane;

    double m_buffer_RT[ 7 ] = { 0, 0, 0, 1, 0, 0, 0 };

    Eigen::Map<Eigen::Quaterniond> m_odometry_R = Eigen::Map<Eigen::Quaterniond>( m_buffer_RT );
    Eigen::Map<Eigen::Vector3d>    m_odometry_T = Eigen::Map<Eigen::Vector3d>( m_buffer_RT + 4 );

    bool m_has_pc_corner = 0;
    bool m_has_pc_full = 0;
    bool m_has_pc_plane = 0;
    bool m_has_odom_message = 0;

    Data_pair()
    {
        m_odometry_R.setIdentity();
        m_odometry_T.setZero();
    }

    void add_pc_corner( Loam_livox_custom_point_cloud &ros_pc );
    void add_pc_plane( Loam_livox_custom_point_cloud &ros_pc );
    void add_pc_full( Loam_livox_custom_point_cloud &ros_pc );
    void add_odom( const nav_msgs::Odometry::ConstPtr &laserOdometry );
    bool is_completed();
};

#endif