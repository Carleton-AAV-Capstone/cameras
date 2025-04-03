# Cameras
Code to publish camera feeds and camera infos to ROS2

## Setup Instructions for humble
```bash
cd ~
mkdir ros2_ws
cd ros2_ws
mkdir src
cd src
git clone https://github.com/ros-perception/image_pipeline.git -b humble
git clone https://github.com/ptrmu/ros2_shared.git
git clone https://github.com/Carleton-AAV-Capstone/cameras
cd ..
colcon build
source install/setup.bash
```

## Linux ~/.bashrc changes
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Executable
```bash
Check right and left cameras with
ls /dev/video*
ffplay /dev/video{#} e.g. /dev/video2

ros2 run cameras camera_feeds_with_info --left_device=/dev/video{#} --right_device=/dev/video{#}
```

## Calibrating New Cameras

Detailed parameters and tutorial can be found here:
https://docs.ros.org/en/rolling/p/camera_calibration/doc/index.html
https://docs.ros.org/en/rolling/p/camera_calibration/doc/tutorial_stereo.html

Note:\
Current code can be improved to take the entire calibration file but currently it is split into left and right where left.ini is everything after the first but before the second\
```#oST version 5.0 parameters```
and right.ini is everything after the second

```bash
ros2 run camera_calibration cameracalibrator \
  --size 9x6 \
  --square 0.04 \
  --approximate 0.3 \
  --ros-args \
  --remap /left:=/left/image_raw \
  --remap /right:=/right/image_raw
```

Replace the left.ini and right.ini in cameras folder before colcon build

## Create rectified, disparity, and pointcloud2 for 2 cameras
```bash
ros2 launch stereo_image_proc stereo_image_proc.launch.py \
approximate_sync:=false \
correlation_window_size:=7 \
disp12_max_diff:=10 \
disparity_range:=256 \
speckle_size:=20 \
stereo_algorithm:=1
```

To test outlines
```bash
ros2 launch stereo_image_proc stereo_image_proc.launch.py \
approximate_sync:=true \
correlation_window_size:=7 \
disp12_max_diff:=10 \
disparity_range:=256 \
min_disparity:=0 \
speckle_range:=3 \
speckle_size:=0 \
stereo_algorithm:=0 \
texture_threshold:=700 \
uniqueness_ratio:=0.0
```

To visualize pointcloud2
```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 4 \
  0 1.5708 1.5708 \
  test_frame_id \
  Test_child_frame_id

rviz2
```