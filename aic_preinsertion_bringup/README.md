# aic_preinsertion_bringup

Combined launch for the qualifying-phase preinsertion stack:

1. Zenoh router (`rmw_zenohd`)
2. Gazebo simulator (`aic_bringup/aic_gz_bringup.launch.py`)
3. Taskboard spawn with extra component configuration
4. Preprocessing, RANSAC/Hough taskboard detection, SC/NIC port detection, and zone projection

## Usage

```bash
source ~/ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 launch aic_preinsertion_bringup aic_preinsertion_bringup.launch.py
```

If a zenoh router is already running:

```bash
ros2 launch aic_preinsertion_bringup aic_preinsertion_bringup.launch.py start_zenoh:=false
```

Gazebo is started the same way as `aic_gz_bringup` (including task-board spawn). Detection is delayed so YOLO does not load while Gazebo is still coming up.


## Simulator / task-board config

Edit `config/simulator.yaml` to change robot pose, cable spawn, task-board pose, and which SC ports, NIC mounts, and connector rails are present.

To use a different file without editing the package copy:

```bash
ros2 launch aic_preinsertion_bringup aic_preinsertion_bringup.launch.py \
  simulator_config:=/path/to/simulator.yaml
```

`aic_gz_bringup` is included with `spawn_task_board:=true` and the task-board component arguments from `simulator.yaml`. Extra launch arguments (SC port / NIC mount presence, translation, orientation) are visible to `spawn_task_board.launch.py` the same way they are when you pass them on the `aic_gz_bringup` command line.


## Detection launch arguments

| Argument | Default | Description |
| --- | --- | --- |
| `detection_startup_delay` | `10.0` | Seconds before detection nodes start |
| `use_sim_time` | `true` | Use Gazebo clock in detection nodes |
| `debug_viz` | `true` | Publish debug image/visualization topics |
| `continue_detecting` | `false` | Keep detecting the taskboard after the first pose |
| `camera_name` | `center` | Camera for SC/NIC detection and zone projection |
| `sc_method` | `yolo` | SC port detector (`yolo` or `hsv`) |
