# Decentralized_loam
## A decentralized framework for simultaneous calibration, localization and mapping with multiple LiDARs

# Introduction
**Decentralized_loam** is the code implementation of our paper "A decentralized framework for simultaneous calibration, localization and mapping with multiple LiDARs", which is developed based on our previous work [Loam-livox](https://github.com/hku-mars/loam_livox). Our project fuses data from multiple LiDARs in a decentralized framework, which can not only address the problem of localization and mapping, but can also online calibrate the extrinsic of 6-DoF (includes 3-DoF of rotation and 3-DoF of translation). 


<div align="center">
    <img src="./pics/maps_align.png" width = 100% >
    <font color=#a0a0a0 size=2>(A). The bird's eye-view of the maps we reconstructed in one of our experiments. The point cloud data sampled from different LiDARs are rendered with different colors. (B). The satellite image of the experiment test ground; (C~E). The detailed inspection of the area marked in dashed circle in A.</font>
</div>

**Developer:** [Jiarong Lin](https://github.com/ziv-lin)

**Our related paper**: our paper is accepted to IROS 2020 and is now available on Arxiv:  
[A decentralized framework for simultaneous calibration, localization and mapping with multiple LiDARs](https://arxiv.org/abs/2007.01483)

**Our related video**: our related videos are now available on YouTube (click below image to open):
<div align="center">
<a href="https://youtu.be/n8r4ch8h8oo" target="_blank"><img src="./pics/title.png" alt="video" width="60%" /></a>
</div>

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation) and its additional ROS pacakge:

```
    sudo apt-get install ros-XXX-cv-bridge ros-XXX-tf ros-XXX-message-filters ros-XXX-image-transport
```
**NOTICE:** remember to replace "XXX" on above command as your ROS distributions, for example, if your use ROS-kinetic, the command should be:

```
    sudo apt-get install ros-kinetic-cv-bridge ros-kinetic-tf ros-kinetic-message-filters ros-kinetic-image-transport
```
### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **PCL**
Our code run abnormally with the lower version of PCL (version < PCL-1.9.0 ) due to some unknown reasons (maybe some bugs), therefore we recommend you install the latest version of [PCL](https://github.com/PointCloudLibrary/pcl)
```
cd YOUR_PATH
git clone https://github.com/PointCloudLibrary/pcl
cd pcl
mkdir build
cd build
cmake ..
sudo make install -j8
```
## 2. Build
Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/hku-mars/decentralized_loam
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Run our example
Download [Our recorded rosbag](https://drive.google.com/file/d/15Sk8vF8Qo8SGR-05aLLICKJ-I1vdksZL/view?usp=sharing) and then
```
roslaunch dc_loam demo.launch
rosbag play dc_loam_demo.bag
```

# Our hardware design
Not only our codes and algorithms are of open-source, but also our hardware design. You can visit this project ([https://github.com/hku-mars/lidar_car_platfrom](https://github.com/hku-mars/lidar_car_platfrom)) for more details :)

<div align="center">
    <a href="https://github.com/hku-mars/lidar_car_platfrom" target="_blank"><img src="./pics/1.png" width = 45% >
    <a href="https://github.com/hku-mars/lidar_car_platfrom" target="_blank"><img src="./pics/2.png" width = 45% >
</div>
<div align="center">
    <img src="./pics/front.jpg" width = 45% >
    <img src="./pics/back.jpg" width = 45% ></a>
</div>


# License
The source code is released under [GPLv2](http://www.gnu.org/licenses/) license.

We are still working on improving the performance and reliability of our codes. For any technical issues, please contact me via email Jiarong Lin < ziv.lin.ljr@gmail.com >.

For commercial use, please contact Dr. Fu Zhang < fuzhang@hku.hk >
