// Author: Jiarong Lin          ziv.lin.ljr@gmail.com

#ifndef __SCENCE_ALIGNMENT_HPP__
#define __SCENCE_ALIGNMENT_HPP__
#include "cell_map_keyframe.hpp"
#include "common_tools.h"
#include "point_cloud_registration.hpp"
#include <iostream>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <stdio.h>
#include <vector>

#include "ceres/ceres.h"
#include "ceres_pose_graph_3d.hpp"
#include "pcl_tools.hpp"
#include "tools_json.hpp"

// ANCHOR Scene_alignment
template <typename PT_DATA_TYPE>
class Scene_alignment
{
  public:
    Common_tools::File_logger file_logger_common, file_logger_timer;
    Common_tools::Timer       timer;

    float                     m_line_res = 0.4;
    float                     m_plane_res = 0.4;
    pcl::VoxelGrid<PointType> m_down_sample_filter_line_source, m_down_sample_filter_line_target;
    pcl::VoxelGrid<PointType> m_down_sample_filter_surface_source, m_down_sample_filter_surface_target;
    int                       pair_idx = 0;
    Point_cloud_registration  m_pc_reg;
    std::string               m_save_path;
    int                       m_para_scene_alignments_maximum_residual_block = 5000;
    int                       m_maximum_icp_iteration = 10;
    float                     m_accepted_threshold = 0.2;
    
    ADD_SCREEN_PRINTF_OUT_METHOD;

    Scene_alignment()
    {
        pair_idx = 0;
        set_downsample_resolution( m_line_res, m_plane_res );
        m_if_verbose_screen_printf = 1;
    }

    Scene_alignment( std::string path )
    {
        init( path );
    };

    ~Scene_alignment()
    {
        file_logger_common.release();
        file_logger_timer.release();
    };

    void set_downsample_resolution( const float &line_res, const float &plane_res );
    void init( std::string path );
    void dump_file_name( std::string                 save_file_name,
                         std::map<int, std::string> &map_filename );
    int  find_tranfrom_of_two_mappings( std::shared_ptr<Maps_keyframe<PT_DATA_TYPE>> keyframe_a,
                                        std::shared_ptr<Maps_keyframe<PT_DATA_TYPE>> keyframe_b,
                                        int                                          if_save = 1,
                                        std::string                                  mapping_save_path = std::string( " " ) );
    int  find_tranfrom_of_two_mappings( Maps_keyframe<PT_DATA_TYPE> *keyframe_a,
                                        Maps_keyframe<PT_DATA_TYPE> *keyframe_b,
                                        int                          if_save = 1,
                                        std::string                  mapping_save_path = std::string( " " ), 
                                        int if_all_plane = 0 );
                                        
    static int load_pose_and_regerror( std::string                               file_name,
                                       Eigen::Quaterniond &                      q_curr,
                                       Eigen::Vector3d &                         t_curr,
                                       Eigen::Matrix<double, Eigen::Dynamic, 1> &mat_reg_err )
    {

        FILE *fp = fopen( file_name.c_str(), "r" );
        if ( fp == nullptr )
        {
            cout << "load_mapping_from_file: " << file_name << " fail!" << std::endl;
            return 0;
        }
        else
        {
            char                      readBuffer[ 1 << 16 ];
            rapidjson::FileReadStream is( fp, readBuffer, sizeof( readBuffer ) );
            rapidjson::Document       doc;

            doc.ParseStream( is );
            if ( doc.HasParseError() )
            {
                printf( "GetParseError, err_code =  %d\n", doc.GetParseError() );
                return 0;
            }
            auto json_arrray = Common_tools::get_json_array<double>( doc[ "Q" ].GetArray() );
            q_curr.w() = json_arrray[ 0 ];
            q_curr.x() = json_arrray[ 1 ];
            q_curr.y() = json_arrray[ 2 ];
            q_curr.z() = json_arrray[ 3 ];

            t_curr = Eigen::Vector3d( Common_tools::get_json_array<double>( doc[ "T" ].GetArray() ) );

            rapidjson::Document::Array json_array = doc[ "Reg_err" ].GetArray();
            size_t                     reg_err_size = json_array.Size();
            mat_reg_err.resize( reg_err_size, 1 );
            for ( size_t i = 0; i < reg_err_size; i++ )
            {
                mat_reg_err( i ) = json_array[ i ].GetDouble();
            }

            return 1;
        }
    }

    static inline Ceres_pose_graph_3d::Constraint3d add_constrain_of_loop( int s_idx, int t_idx,
                                                                           Eigen::Quaterniond q_a, Eigen::Vector3d t_a,
                                                                           Eigen::Quaterniond q_b, Eigen::Vector3d t_b,
                                                                           Eigen::Quaterniond icp_q, Eigen::Vector3d icp_t,
                                                                           int if_verbose = 1 )
    {
        Ceres_pose_graph_3d::Constraint3d pose_constrain;
        auto                              q_res = q_b.inverse() * icp_q.inverse() * q_a;
        //q_res = q_res.inverse();
        auto t_res = q_b.inverse() * ( icp_q.inverse() * ( t_a - icp_t ) - t_b );
        //t_res = q_res.inverse()*(-t_res);
        //q_res = q_res.inverse();
        if ( if_verbose == 0 )
        {
            cout << "=== Add_constrain_of_loop ====" << endl;
            cout << q_a.coeffs().transpose() << endl;
            cout << q_b.coeffs().transpose() << endl;
            cout << icp_q.coeffs().transpose() << endl;
            cout << t_a.transpose() << endl;
            cout << t_b.transpose() << endl;
            cout << icp_t.transpose() << endl;
            cout << "Result: " << endl;
            cout << q_res.coeffs().transpose() << endl;
            cout << t_res.transpose() << endl;
        }
        //t_res.setZero();
        pose_constrain.id_begin = s_idx;
        pose_constrain.id_end = t_idx;
        pose_constrain.t_be.p = t_res;
        pose_constrain.t_be.q = q_res;

        return pose_constrain;
    };

    static inline void save_edge_and_vertex_to_g2o( std::string                               file_name,
                                                    Ceres_pose_graph_3d::MapOfPoses &         pose3d_map,
                                                    Ceres_pose_graph_3d::VectorOfConstraints &pose_csn_vec )
    {
        FILE *fp = fopen( file_name.c_str(), "w+" );
        if ( fp != NULL )
        {
            cout << "Dump to g2o files:" << file_name << std::endl;
            for ( auto it = pose3d_map.begin(); it != pose3d_map.end(); it++ )
            {
                Ceres_pose_graph_3d::Pose3d pose3d = it->second;
                fprintf( fp, "VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f\n", ( int ) it->first,
                         pose3d.p( 0 ), pose3d.p( 1 ), pose3d.p( 2 ),
                         pose3d.q.x(), pose3d.q.y(), pose3d.q.z(), pose3d.q.w() );
            }
            for ( size_t i = 0; i < pose_csn_vec.size(); i++ )
            {
                auto csn = pose_csn_vec[ i ];
                fprintf( fp, "EDGE_SE3:QUAT %d %d %f %f %f %f %f %f %f", csn.id_begin, csn.id_end,
                         csn.t_be.p( 0 ), csn.t_be.p( 1 ), csn.t_be.p( 2 ),
                         csn.t_be.q.x(), csn.t_be.q.y(), csn.t_be.q.z(), csn.t_be.q.w() );
                Eigen::Matrix<double, 6, 6> info_mat;
                info_mat.setIdentity();
                //info_mat *= abs(csn.id_end - csn.id_begin);
                for ( size_t c = 0; c < 6; c++ )
                {
                    for ( size_t r = c; r < 6; r++ )
                    {
                        fprintf( fp, " %f", info_mat( c, r ) );
                    }
                }
                fprintf( fp, "\n" );
            }
            fclose( fp );
            cout << "Dump to g2o file OK, file name: " << file_name << std::endl;
        }
        else
        {
            cout << "Open file name " << file_name << " error, please check" << endl;
        }
    };

    static inline void save_edge_and_vertex_to_g2o( std::string                               file_name,
                                                    Ceres_pose_graph_3d::VectorOfPose         pose3d_vec,
                                                    Ceres_pose_graph_3d::VectorOfConstraints &pose_csn_vec )
    {
        FILE *fp = fopen( file_name.c_str(), "w+" );
        if ( fp != NULL )
        {
            cout << "Dump to g2o files:" << file_name << std::endl;
            for ( size_t i = 0; i < pose3d_vec.size(); i++ )
            {
                fprintf( fp, "VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f\n", ( int ) i,
                         pose3d_vec[ i ].p( 0 ), pose3d_vec[ i ].p( 1 ), pose3d_vec[ i ].p( 2 ),
                         pose3d_vec[ i ].q.x(), pose3d_vec[ i ].q.y(), pose3d_vec[ i ].q.z(), pose3d_vec[ i ].q.w() );
            }
            for ( size_t i = 0; i < pose_csn_vec.size(); i++ )
            {
                auto csn = pose_csn_vec[ i ];
                fprintf( fp, "EDGE_SE3:QUAT %d %d %f %f %f %f %f %f %f", csn.id_begin, csn.id_end,
                         csn.t_be.p( 0 ), csn.t_be.p( 1 ), csn.t_be.p( 2 ),
                         csn.t_be.q.x(), csn.t_be.q.y(), csn.t_be.q.z(), csn.t_be.q.w() );
                Eigen::Matrix<double, 6, 6> info_mat;
                info_mat.setIdentity();
                for ( size_t c = 0; c < 6; c++ )
                {
                    for ( size_t r = c; r < 6; r++ )
                    {
                        fprintf( fp, " %f", info_mat( c, r ) );
                    }
                }
                fprintf( fp, "\n" );
            }
            fclose( fp );
            cout << "Dump to g2o file OK, file name: " << file_name << std::endl;
        }
        else
        {
            cout << "Open file name " << file_name << " error, please check" << endl;
        }
    };

};

#endif
