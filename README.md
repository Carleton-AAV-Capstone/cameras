# Cameras
Code to publish camera feed to ROS2

# Setup Instructions
If using pycharm in linux
settings -> Project -> interpreter -> select interpreter -> show all -> show interpreter paths
add the following:
```
/opt/ros/humble/lib/python3.10/site-packages
/opt/ros/humble/local/lib/python3.10/dist-packages
```

This will let ros2 packages be understood by python in pycharm

linux ~/.bashrc changes
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-11.5/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/opt/ros/humble/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export WORKON_HOME=\$HOME/.virtualenvs" >> ~/.bashrc
echo "export PROJECT_HOME=\$HOME/Devel" >> ~/.bashrc
echo "export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:\$PYTHONPATH" >> ~/.bashrc
echo "export AMENT_PREFIX_PATH=/opt/ros/humble:\$AMENT_PREFIX_PATH" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
echo "export ROS_DISTRO=humble" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc

source ~/.bashrc
```

