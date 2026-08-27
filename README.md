# AIC Preinsertion strategy


## Preliminaries

This code was tested on an Ubuntu 24.04 WSL on a Windows 11 machine with a Nvidia RTX 5090

It included the following in in the end of bashrc:

```
# GPU wsl2 handling
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export GALLIUM_DRIVER=d3d12
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

## ROS2 
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
source ~/ws_aic/install/setup.bash
```

Then install the following projects:

* Install [ROS kilted full desktop](https://docs.ros.org/en/kilted/Installation.html) 
* Install the AIC toolbox (https://github.com/intrinsic-dev/aic) with the [source installation instructions](https://github.com/intrinsic-dev/aic/blob/main/docs/build_eval.md).

## Disclaimer

This repository is developed with help of [Cursor](https://cursor.com/), mostly using Grok 4.6 as auto-selected model



## Install this repository

```
git clone https://github.com/knmcguire/aic_taskboard_detection_qualifying_phase
GZ_BUILD_FROM_SOURCE=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --merge-install --symlink-install --packages-ignore lerobot_robot_aic --packages-select aic_component_detection aic_preinsertion_control aic_taskboard_detection, aic_preinsertion_bringup 

```

## Tryout the default strategy


Combined bringup (zenoh, Gazebo, preprocessing, taskboard detection, SC/NIC detection, zone projection):

```
ros2 launch aic_preinsertion_bringup aic_preinsertion_bringup.launch.py
```

Task-board spawn is configured in `aic_preinsertion_bringup/config/simulator.yaml`.

Or start the pieces one by one:

Startup a zenoh router

```
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Startup the simulation

```
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true start_aic_engine:=false spawn_task_board:=true sc_port_0_present:=true sc_port_1_present:=true nic_card_mount_4_present:=true nic_card_mount_2_present:=true nic_card_mount_1_present:=true 
```

Make the blob detection and canny detection

```
ros2 launch aic_taskboard_detection preprocessing.launch.xml
```


Startup the taskboard detection
```
ros2 launch aic_taskboard_detection ransac_hough_taskboard_detection.launch.xml
```


Startup taskboard component detection
```
ros2 launch aic_component_detection component_detection.launch.xml debug_viz:=true
```

## Controlling the arm

With the component detection active you can either move the arm using your keyboard

```
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

or try out the automatic control strategy 

```
ros2 launch aic_preinsertion_control preinsertion_control.launch.xml
```

