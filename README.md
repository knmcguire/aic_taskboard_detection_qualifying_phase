

Startup a zenoh router

```
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Startup the simulation

```
ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true start_aic_engine:=false spawn_task_board:=true
```

Make the blob detection and canny detection

```
ros2 launch aic_taskboard_detection preprocessing.launch.xml
```


Startup the taskboard detection
```
ros2 launch aic_taskboard_detection ransac_hough_taskboard_detection.launch.xml
```