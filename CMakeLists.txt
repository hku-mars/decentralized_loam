cmake_minimum_required(VERSION 2.8)
project(dc_loam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3")
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

find_package(PCL REQUIRED)
message(STATUS "***** PCL version: ${PCL_VERSION} *****")
message(STATUS "***** Eigen version: ${EIGEN_VERSION} *****")
####

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ./include/tools/
  ./source/
  )

add_message_files(
  DIRECTORY msg
  FILES
  Loam_livox_custom_point_cloud.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime message_generation
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)


add_executable(laser_scan2scan source/laser_scan2scan.cpp)
target_link_libraries(laser_scan2scan ${catkin_LIBRARIES} ${PCL_LIBRARIES}  ${CERES_LIBRARIES})

add_executable(laser_feature_extraction source/laser_feature_extractor.cpp)
target_link_libraries(laser_feature_extraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(laser_mapping source/laser_mapping.cpp
                              source/lidar_agent.cpp
                              source/point_cloud_registration.cpp
                              source/scene_alignment.cpp
                              source/mapping_process.cpp
                              source/mapping_pointcloud_cell.cpp
                              source/mapping_pointcloud_keyframe.cpp
                              source/mapping_pointcloud_map.cpp
                              source/mapping_service.cpp
                              source/ekf_pose_6d.cpp )

target_link_libraries(laser_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} )
set_property(TARGET laser_mapping PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(read_camera source/read_camera.cpp)
target_link_libraries(read_camera ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(ros_test source/ros_test.cpp)
target_link_libraries(ros_test ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} )

# target_precompile_headers(laser_mapping
#   PUBLIC
#     source/laser_mapping.hpp
# )
