"""Combined preinsertion bringup: zenoh, Gazebo, and detection pipeline."""

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.launch_description_sources import (
    FrontendLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Startup lines the stages wait for, rather than guessing at timeouts.
ZENOH_READY = "Started Zenoh router"
TASKBOARD_DETECTION_READY = "RANSAC/Hough taskboard detection started for"

POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
COMPONENT_FIELDS = ("present", "translation", "roll", "pitch", "yaw")
TASK_BOARD_COMPONENTS = (
    "lc_mount_rail_0",
    "sfp_mount_rail_0",
    "sc_mount_rail_0",
    "lc_mount_rail_1",
    "sfp_mount_rail_1",
    "sc_mount_rail_1",
    "sc_port_0",
    "sc_port_1",
    "sc_port_2",
    "sc_port_3",
    "sc_port_4",
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
)


def _to_launch_str(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _pose_args(prefix, pose):
    args = {}
    if not pose:
        return args
    for axis in POSE_AXES:
        if axis in pose:
            args[f"{prefix}_{axis}"] = _to_launch_str(pose[axis])
    return args


def _task_board_spawn_args(task_board):
    args = _pose_args("task_board", task_board.get("pose", {}))
    components = task_board.get("components", {})
    for name in TASK_BOARD_COMPONENTS:
        fields = components.get(name, {})
        for field in COMPONENT_FIELDS:
            if field in fields:
                args[f"{name}_{field}"] = _to_launch_str(fields[field])
    return args


def _gate(actions):
    """Hold back `actions` until a trigger releases them, which happens once."""
    pending = list(actions)

    def release():
        released, pending[:] = list(pending), []
        return released or None

    return release


def _on_output(pattern, release):
    """Release a gate when any running process prints `pattern`."""

    def handler(event):
        if pattern in event.text.decode(errors="replace"):
            return release()
        return None

    return RegisterEventHandler(OnProcessIO(on_stdout=handler, on_stderr=handler))


def _on_clean_exit(matches, release):
    """Release a gate when a process matching `matches` exits successfully."""

    def handler(event, context):
        # Processes also exit cleanly while the stack is being torn down.
        if context.is_shutdown:
            return None
        if event.returncode == 0 and matches(event):
            return release()
        return None

    return RegisterEventHandler(OnProcessExit(on_exit=handler))


def _spawner_of(controller):
    def matches(event):
        cmd = [str(part) for part in event.cmd]
        return cmd[0].endswith("spawner") and controller in cmd

    return matches


def _scoped_include(include):
    return GroupAction(actions=[include], scoped=True)


def _load_simulator_config(config_path):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Simulator config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def launch_setup(context, *args, **kwargs):
    cfg = _load_simulator_config(
        LaunchConfiguration("simulator_config").perform(context)
    )
    sim = cfg.get("simulator", {})
    task_board = cfg.get("task_board", {})

    start_zenoh = _as_bool(
        LaunchConfiguration("start_zenoh").perform(context), default=True
    )
    initial_joint_controller = LaunchConfiguration(
        "initial_joint_controller"
    ).perform(context)

    gz_launch_args = {
        "initial_joint_controller": initial_joint_controller,
        "ground_truth": _to_launch_str(sim.get("ground_truth", True)),
        "start_aic_engine": _to_launch_str(sim.get("start_aic_engine", False)),
        "gazebo_gui": _to_launch_str(sim.get("gazebo_gui", True)),
        "launch_rviz": _to_launch_str(sim.get("launch_rviz", True)),
        "spawn_task_board": _to_launch_str(task_board.get("spawn", True)),
        "spawn_cable": _to_launch_str(sim.get("spawn_cable", False)),
        "attach_cable_to_gripper": _to_launch_str(
            sim.get("attach_cable_to_gripper", False)
        ),
        "cable_type": str(sim.get("cable_type", "sfp_sc_cable")),
    }
    gz_launch_args.update(_pose_args("robot", sim.get("robot", {})))
    gz_launch_args.update(_pose_args("cable", sim.get("cable", {})))
    gz_launch_args.update(_task_board_spawn_args(task_board))

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("aic_bringup"), "launch", "aic_gz_bringup.launch.py"]
            )
        ),
        launch_arguments=list(gz_launch_args.items()),
    )

    detection_args = {
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "debug_viz": LaunchConfiguration("debug_viz"),
    }

    preprocessing = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aic_taskboard_detection"),
                    "launch",
                    "preprocessing.launch.xml",
                ]
            )
        ),
        launch_arguments={
            "params_file": PathJoinSubstitution(
                [
                    FindPackageShare("aic_taskboard_detection"),
                    "config",
                    "preprocessing.params.yaml",
                ]
            ),
        }.items(),
    )

    taskboard_detection = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aic_taskboard_detection"),
                    "launch",
                    "ransac_hough_taskboard_detection.launch.xml",
                ]
            )
        ),
        launch_arguments=list(
            {
                **detection_args,
                "params_file": PathJoinSubstitution(
                    [
                        FindPackageShare("aic_taskboard_detection"),
                        "config",
                        "ransac_hough_taskboard_detection.params.yaml",
                    ]
                ),
                "fusion_params_file": PathJoinSubstitution(
                    [
                        FindPackageShare("aic_taskboard_detection"),
                        "config",
                        "taskboard_tf_fusion.params.yaml",
                    ]
                ),
                "continue_detecting": LaunchConfiguration("continue_detecting"),
            }.items()
        ),
    )

    component_detection = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aic_component_detection"),
                    "launch",
                    "component_detection.launch.xml",
                ]
            )
        ),
        launch_arguments=list(
            {
                **detection_args,
                "params_file": PathJoinSubstitution(
                    [
                        FindPackageShare("aic_component_detection"),
                        "config",
                        "component_detection.params.yaml",
                    ]
                ),
                "camera_name": LaunchConfiguration("camera_name"),
                "sc_method": LaunchConfiguration("sc_method"),
            }.items()
        ),
    )

    zenoh_router = Node(
        package="rmw_zenoh_cpp",
        executable="rmw_zenohd",
        name="rmw_zenohd",
        output="screen",
    )

    # Each stage waits for the previous one to report itself ready. Handlers are
    # registered before anything starts so no event can be missed.
    actions = [
        # The arm controller activating is the point where Gazebo has finished
        # loading and ros2_control is running.
        _on_clean_exit(
            _spawner_of(initial_joint_controller),
            _gate([_scoped_include(preprocessing), _scoped_include(taskboard_detection)]),
        ),
        _on_output(
            TASKBOARD_DETECTION_READY,
            _gate([_scoped_include(component_detection)]),
        ),
    ]
    if start_zenoh:
        actions.append(zenoh_router)
        actions.append(_on_output(ZENOH_READY, _gate([simulator])))
    else:
        actions.append(simulator)
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value="rmw_zenoh_cpp"),
            DeclareLaunchArgument(
                "simulator_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("aic_preinsertion_bringup"),
                        "config",
                        "simulator.yaml",
                    ]
                ),
                description="YAML file with Gazebo and task-board spawn settings.",
            ),
            DeclareLaunchArgument(
                "start_zenoh",
                default_value="true",
                description="Start the rmw_zenohd router before the rest of the stack.",
            ),
            DeclareLaunchArgument(
                "initial_joint_controller",
                default_value="aic_controller",
                description="Controller Gazebo activates last; its spawner exiting starts detection.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation time in detection nodes.",
            ),
            DeclareLaunchArgument(
                "debug_viz",
                default_value="true",
                description="Publish detection debug visualizations.",
            ),
            DeclareLaunchArgument(
                "continue_detecting",
                default_value="false",
                description="Keep running taskboard detection after the first successful pose.",
            ),
            DeclareLaunchArgument(
                "camera_name",
                default_value="center",
                description="Camera used by SC/NIC port detection and zone projection.",
            ),
            DeclareLaunchArgument(
                "sc_method",
                default_value="yolo",
                description="SC port detection method (yolo or hsv).",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
