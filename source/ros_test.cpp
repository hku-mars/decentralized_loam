#include "custom_point_cloud_interface.hpp"

Loam_livox_custom_point_cloud custom_pc_msg;
Loam_livox_custom_point_cloud custom_pc_msg_a, custom_pc_msg_b;

using std::cout;
using std::endl;

ADD_SCREEN_PRINTF_OUT_METHOD;
Custom_point_cloud_interface ctm_pc_itf;

void pc_subscriber( Loam_livox_custom_point_cloud custom_pc_msg )
{
    cout << "Receive message, current msg time: " << custom_pc_msg.header.stamp.toSec() << endl;
    cout << "Point size = " << custom_pc_msg.m_point_size << endl;
}

int main( int argc, char **argv )
{

    m_if_verbose_screen_printf = 0;
    printf_program( "This is rostest" );
    ros::init( argc, argv, "ros_test" );
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher  pub;
    sub = nh.subscribe<Loam_livox_custom_point_cloud>( "ros_test_pub", 1000, pc_subscriber );
    pub = nh.advertise<Loam_livox_custom_point_cloud>( "ros_test_pub", 1000 );
    custom_pc_msg.header.stamp = ros::Time::now();

    size_t pt_size = 1;

    custom_pc_msg.m_point_size = 10;

    custom_pc_msg.m_point_xyz.resize( custom_pc_msg.m_point_size * 4 );
    custom_pc_msg.m_intensity.resize( custom_pc_msg.m_point_size );
    custom_pc_msg.m_timestamp.resize( custom_pc_msg.m_point_size );
    auto ptr = std::make_shared<Loam_livox_custom_point_cloud>( custom_pc_msg );
    ctm_pc_itf.set_msg_ptr( ptr );
    pcl::PointCloud<pcl::PointXYZI> pcl_xyzi;
    Custom_point_cloud_interface::msg_to_pcl_pc( custom_pc_msg, pcl_xyzi );
    Custom_point_cloud_interface::msg_from_pcl_pc( custom_pc_msg, pcl_xyzi );
    screen_out << "Size of Header : " << sizeof( custom_pc_msg.header ) << endl;
    screen_out << "Size of Custom_message : " << sizeof( custom_pc_msg ) << endl;
    custom_pc_msg_a = custom_pc_msg;
    custom_pc_msg_b = custom_pc_msg;
    custom_pc_msg_a = custom_pc_msg_a + custom_pc_msg_b;
    cout << "Size of a+b: " << custom_pc_msg_a.m_point_size << endl;
    cout << "Size of a+b: " << custom_pc_msg_a.m_point_xyz.size() << endl;
    for ( size_t i = 0; i < pt_size; i++ )
    {
    }

    for ( size_t i = 0; i < 2; i++ )
    {
        pub.publish( custom_pc_msg );
        ros::spinOnce();
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    printf( "End of program\r\n" );
}