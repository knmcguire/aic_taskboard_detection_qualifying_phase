

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