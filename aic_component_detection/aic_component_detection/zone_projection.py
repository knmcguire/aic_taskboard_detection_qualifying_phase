#!/usr/bin/env python3

"""Project NIC and SC port detections onto the taskboard zone planes.

Inputs:
  - TF ``taskboard_detected``
  - NIC port pixel centers and 2D detections
  - SC port 2D detections

Each image-pixel detection is cast as a camera ray, intersected with the
elevated zone plane in the taskboard frame, then labeled by rail (and, for
NIC ports, by port index 0/1). Zone 1 holds NIC cards; zone 2 holds SC ports.

Publishes child TFs of ``taskboard_detected`` on ``/tf``:
  ``nic_port_r<rail>_p<port>`` and ``sc_port_r<rail>``.
Optional latched entrance frames (``nic_port_entrance_*``, ``sc_port_entrance_*``)
are also published dynamically so ``reset_zone_monitoring`` can drop them.
"""

from argparse import ArgumentParser

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

# Zone map in the taskboard XY plane (origin at board center):
#   1 = bottom-left (NIC), 2 = top-left (SC), 3 = bottom-right, 4 = top-right.
NIC_ZONE = 1
SC_ZONE = 2
NIC_ZONE_COLOR = (30, 180, 255)
SC_ZONE_COLOR = (180, 80, 255)
RAIL_COLOR = (120, 255, 120)
NIC_POINT_COLOR = (0, 255, 0)
SC_POINT_COLOR = (0, 220, 255)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


class ZoneProjection(Node):
    """Ray-cast component detections onto taskboard zones and publish port TFs."""

    def __init__(self):
        super().__init__('zone_projection')

        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        self.short_edge_m = float(self.declare_parameter('taskboard_short_edge_m', 0.30).value)
        self.long_edge_m = float(self.declare_parameter('taskboard_long_edge_m', 0.42).value)
        self.nic_zone_offset_m = float(self.declare_parameter('nic_zone_offset_m', 0.11).value)
        self.sc_zone_offset_m = float(self.declare_parameter('sc_zone_offset_m', 0.11).value)
        self.rail_count = max(1, int(self.declare_parameter('zone_rail_count', 5).value))
        self.sc_active_rail_count = max(
            1, int(self.declare_parameter('sc_active_rail_count', 2).value)
        )
        self.sc_active_rail_count = min(self.sc_active_rail_count, self.rail_count)

        default_image_topic = f'/{self.camera_name}_camera/image'
        default_camera_info_topic = f'/{self.camera_name}_camera/camera_info'
        default_camera_frame = f'{self.camera_name}_camera/optical'
        default_viz_topic = f'/{self.camera_name}_camera/image_zone_projection'

        self.input_image_topic = str(
            self.declare_parameter('input_image_topic', default_image_topic).value
        )
        self.input_camera_info_topic = str(
            self.declare_parameter('input_camera_info_topic', default_camera_info_topic).value
        )
        self.output_image_topic = str(
            self.declare_parameter('output_image_topic', default_viz_topic).value
        )
        self.nic_port_pixel_topic = str(
            self.declare_parameter(
                'nic_port_pixel_topic', '/nic_port_detection/port_pixel_centers'
            ).value
        )
        self.nic_detections_topic = str(
            self.declare_parameter('nic_detections_topic', '/nic_port_detection/detections').value
        )
        self.sc_detections_topic = str(
            self.declare_parameter('sc_detections_topic', '/sc_port_detection/detections').value
        )
        self.publish_debug_visualization = _as_bool(
            self.declare_parameter('publish_debug_visualization', True).value
        ) or _as_bool(
            self.declare_parameter('publish_visualization', True).value
        )
        self.publish_port_tfs = _as_bool(self.declare_parameter('publish_port_tfs', True).value)
        self.publish_port_entrance_tfs = _as_bool(
            self.declare_parameter('publish_port_entrance_tfs', False).value
        )
        self.nic_port_tf_prefix = str(
            self.declare_parameter('nic_port_tf_prefix', 'nic_port').value
        ).strip() or 'nic_port'
        self.sc_port_tf_prefix = str(
            self.declare_parameter('sc_port_tf_prefix', 'sc_port').value
        ).strip() or 'sc_port'
        self.nic_port_entrance_tf_prefix = str(
            self.declare_parameter('nic_port_entrance_tf_prefix', 'nic_port_entrance').value
        ).strip()
        self.sc_port_entrance_tf_prefix = str(
            self.declare_parameter('sc_port_entrance_tf_prefix', 'sc_port_entrance').value
        ).strip()
        self.port_entrance_min_consecutive_hits = max(
            1, int(self.declare_parameter('port_entrance_min_consecutive_hits', 2).value)
        )
        self.port_entrance_position_epsilon_m = max(
            0.0, float(self.declare_parameter('port_entrance_position_epsilon_m', 0.005).value)
        )
        self.port_pixel_max_age_sec = float(
            self.declare_parameter('port_pixel_max_age_sec', 0.5).value
        )
        self.nic_detection_association_max_px = float(
            self.declare_parameter('nic_detection_association_max_px', 80.0).value
        )
        self.tf_lookup_timeout_sec = float(self.declare_parameter('tf_lookup_timeout_sec', 0.15).value)
        self.stale_warn_interval_sec = float(
            self.declare_parameter('stale_warn_interval_sec', 2.0).value
        )
        self.rail_overflow_warn_interval_sec = float(
            self.declare_parameter('rail_overflow_warn_interval_sec', 2.0).value
        )
        self.min_projected_depth_m = float(
            self.declare_parameter('min_projected_depth_m', 1e-4).value
        )
        self.overlay_alpha = float(self.declare_parameter('overlay_alpha', 0.25).value)
        self.camera_frame = str(self.declare_parameter('camera_frame', default_camera_frame).value)
        self.taskboard_frame = str(
            self.declare_parameter('taskboard_frame', 'taskboard_detected').value
        )
        self.use_camera_info_frame = bool(self.declare_parameter('use_camera_info_frame', True).value)
        self.prefer_image_timestamp = bool(self.declare_parameter('prefer_image_timestamp', True).value)
        sensor_qos_depth = max(1, int(self.declare_parameter('sensor_qos_depth', 1).value))
        publisher_qos_depth = max(1, int(self.declare_parameter('publisher_qos_depth', 10).value))

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.latest_nic_port_pixels_msg = None
        self.latest_nic_detections_msg = None
        self.latest_sc_detections_msg = None
        self.latest_nic_port_pixels_received_time = None
        self.latest_sc_detections_received_time = None
        self._last_stale_port_warn_time = None
        self._last_stale_sc_warn_time = None
        self._last_overflow_port_warn_time = None
        self._port_entrance_candidate_positions = {}
        self._port_entrance_candidate_hits = {}
        self._latched_port_entrance_frames = {}
        self.monitoring_active = False
        self._last_status_log_time = None
        self._cam_pos_tb = None
        self._cam_rot_tb = None
        self._tb_pos_cam = None
        self._tb_rot_cam = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=sensor_qos_depth,
        )
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, self.input_image_topic, self.image_callback, sensor_qos)
        self.create_subscription(
            CameraInfo, self.input_camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.create_subscription(
            CameraInfo, self.input_camera_info_topic, self.camera_info_callback, camera_info_qos
        )
        self.create_subscription(
            PoseArray, self.nic_port_pixel_topic, self._nic_port_pixels_callback, sensor_qos
        )
        self.create_subscription(
            Detection2DArray, self.nic_detections_topic, self._nic_detections_callback, sensor_qos
        )
        self.create_subscription(
            Detection2DArray, self.sc_detections_topic, self._sc_detections_callback, sensor_qos
        )
        self.image_pub = None
        if self.publish_debug_visualization:
            self.image_pub = self.create_publisher(Image, self.output_image_topic, publisher_qos_depth)

        self.add_on_set_parameters_callback(self._parameter_callback)
        self.create_service(Trigger, 'start_zone_monitoring', self.start_zone_monitoring_callback)
        self.create_service(Trigger, 'check_zone_monitoring', self.check_zone_monitoring_callback)
        self.create_service(Trigger, 'reset_zone_monitoring', self.reset_zone_monitoring_callback)
        self.create_timer(0.5, self._activation_timer)

        self.get_logger().info('Zone projection node started.')
        self.get_logger().info(f'Camera: {self.camera_name}  frame: {self.camera_frame}')
        self.get_logger().info(f'Taskboard frame: {self.taskboard_frame}')
        self.get_logger().info(f'Subscribing image: {self.input_image_topic}')
        self.get_logger().info(f'Subscribing camera info: {self.input_camera_info_topic}')
        self.get_logger().info(f'Subscribing NIC port pixels: {self.nic_port_pixel_topic}')
        self.get_logger().info(f'Subscribing NIC detections: {self.nic_detections_topic}')
        self.get_logger().info(f'Subscribing SC detections: {self.sc_detections_topic}')
        self.get_logger().info(
            f'NIC zone {NIC_ZONE} offset={self.nic_zone_offset_m:+.3f}m, '
            f'SC zone {SC_ZONE} offset={self.sc_zone_offset_m:+.3f}m, '
            f'rails={self.rail_count} (SC uses first {self.sc_active_rail_count})'
        )
        self.get_logger().info(
            f'Debug overlay: {"on" if self.publish_debug_visualization else "off"} '
            f'({self.output_image_topic})'
        )
        if self.publish_port_tfs:
            self.get_logger().info(
                f'Publishing port TFs under "{self.taskboard_frame}": '
                f'{self.nic_port_tf_prefix}_r*_p* and {self.sc_port_tf_prefix}_r*'
            )
        if self.publish_port_entrance_tfs:
            self.get_logger().info(
                f'Publishing latched port-entrance TFs on /tf under "{self.taskboard_frame}": '
                f'{self.nic_port_entrance_tf_prefix}_r*_p* and '
                f'{self.sc_port_entrance_tf_prefix}_r* '
                f'(min_hits={self.port_entrance_min_consecutive_hits}, '
                f'epsilon={self.port_entrance_position_epsilon_m:.4f}m)'
            )

    def camera_info_callback(self, msg):
        if self.camera_info_received:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.camera_info_received = True
        if self.use_camera_info_frame and msg.header.frame_id:
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(f'Using camera frame from CameraInfo: {self.camera_frame}')
        self.get_logger().info(
            f'Camera info received: fx={self.camera_matrix[0, 0]:.2f}, '
            f'fy={self.camera_matrix[1, 1]:.2f}'
        )

    def _nic_port_pixels_callback(self, msg):
        self.latest_nic_port_pixels_msg = msg
        self.latest_nic_port_pixels_received_time = self.get_clock().now()

    def _nic_detections_callback(self, msg):
        self.latest_nic_detections_msg = msg

    def _sc_detections_callback(self, msg):
        self.latest_sc_detections_msg = msg
        self.latest_sc_detections_received_time = self.get_clock().now()

    def _throttled_info(self, message):
        now = self.get_clock().now()
        if (
            self._last_status_log_time is None
            or (now - self._last_status_log_time).nanoseconds
            > int(self.stale_warn_interval_sec * 1e9)
        ):
            self.get_logger().info(message)
            self._last_status_log_time = now

    def _taskboard_tf_available(self):
        try:
            return self.tf_buffer.can_transform(
                self.camera_frame,
                self.taskboard_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except Exception:
            return False

    def _activate_if_tf_ready(self):
        if self.monitoring_active:
            return True
        if not self._taskboard_tf_available():
            return False
        self.monitoring_active = True
        self.get_logger().info(
            f'Zone projection activated: TF "{self.taskboard_frame}" is available '
            f'(camera frame "{self.camera_frame}").'
        )
        return True

    def _activation_timer(self):
        if self.monitoring_active:
            return
        if self._activate_if_tf_ready():
            return
        self._throttled_info(
            f'Waiting for TF "{self.camera_frame}" -> "{self.taskboard_frame}" '
            'before projecting zones.'
        )

    def start_zone_monitoring_callback(self, request, response):
        if self._activate_if_tf_ready() or self.monitoring_active:
            response.success = True
            response.message = f'started|{self.taskboard_frame}'
            return response
        response.success = False
        response.message = f'no_tf|taskboard frame "{self.taskboard_frame}" not found'
        self.get_logger().warn(response.message)
        return response

    def check_zone_monitoring_callback(self, request, response):
        if self.monitoring_active:
            response.success = True
            response.message = f'active|{self.taskboard_frame}'
        else:
            response.success = False
            response.message = 'inactive'
        return response

    def reset_zone_monitoring_callback(self, request, response):
        self.monitoring_active = False
        self._port_entrance_candidate_positions.clear()
        self._port_entrance_candidate_hits.clear()
        self._latched_port_entrance_frames.clear()
        response.success = True
        response.message = 'reset_complete'
        self.get_logger().info('Zone monitoring reset; latched port-entrance TFs dropped.')
        return response

    def _parameter_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == 'nic_zone_offset_m' and param.type_ == Parameter.Type.DOUBLE:
                self.nic_zone_offset_m = float(param.value)
            elif param.name == 'sc_zone_offset_m' and param.type_ == Parameter.Type.DOUBLE:
                self.sc_zone_offset_m = float(param.value)
            elif param.name == 'zone_rail_count' and param.type_ == Parameter.Type.INTEGER:
                self.rail_count = max(1, int(param.value))
                self.sc_active_rail_count = min(self.sc_active_rail_count, self.rail_count)
            elif param.name == 'sc_active_rail_count' and param.type_ == Parameter.Type.INTEGER:
                self.sc_active_rail_count = min(max(1, int(param.value)), self.rail_count)
            elif param.name == 'port_pixel_max_age_sec' and param.type_ == Parameter.Type.DOUBLE:
                self.port_pixel_max_age_sec = float(param.value)
            elif param.name == 'publish_port_tfs' and param.type_ == Parameter.Type.BOOL:
                self.publish_port_tfs = bool(param.value)
            elif param.name == 'publish_port_entrance_tfs' and param.type_ == Parameter.Type.BOOL:
                self.publish_port_entrance_tfs = bool(param.value)
                if not self.publish_port_entrance_tfs:
                    self._port_entrance_candidate_positions.clear()
                    self._port_entrance_candidate_hits.clear()
                    self._latched_port_entrance_frames.clear()
        return SetParametersResult(successful=True)

    def _zone_bounds(self, zone_number):
        x_mid = 0.0
        y_mid = 0.0
        x_min_all = -self.short_edge_m * 0.5
        x_max_all = self.short_edge_m * 0.5
        y_min_all = -self.long_edge_m * 0.5
        y_max_all = self.long_edge_m * 0.5
        if zone_number == 1:
            return x_min_all, x_mid, y_min_all, y_mid
        if zone_number == 2:
            return x_min_all, x_mid, y_mid, y_max_all
        if zone_number == 3:
            return x_mid, x_max_all, y_min_all, y_mid
        return x_mid, x_max_all, y_mid, y_max_all

    def _zone_corners(self, zone_number, plane_z):
        x_min, x_max, y_min, y_max = self._zone_bounds(zone_number)
        return np.array(
            [
                [x_min, y_max, plane_z],
                [x_max, y_max, plane_z],
                [x_max, y_min, plane_z],
                [x_min, y_min, plane_z],
            ],
            dtype=np.float32,
        )

    def _zone_rail_boundaries(self, zone_number, plane_z):
        x_min, x_max, y_min, y_max = self._zone_bounds(zone_number)
        rail_span = (y_max - y_min) / float(self.rail_count)
        boundaries = []
        for i in range(1, self.rail_count):
            y_boundary = y_min + i * rail_span
            boundaries.append(
                np.array(
                    [[x_min, y_boundary, plane_z], [x_max, y_boundary, plane_z]],
                    dtype=np.float32,
                )
            )
        return boundaries

    def _zone_rail_index(self, y_value, zone_number):
        _, _, y_min, y_max = self._zone_bounds(zone_number)
        span = max(1e-9, y_max - y_min)
        rail_pos = (float(y_value) - y_min) / span
        rail = int(np.floor(rail_pos * self.rail_count))
        return int(np.clip(rail, 0, self.rail_count - 1))

    def _point_in_zone(self, point, zone_number):
        x_min, x_max, y_min, y_max = self._zone_bounds(zone_number)
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

    @staticmethod
    def quaternion_to_rotation_matrix(quat_xyzw):
        x, y, z, w = quat_xyzw
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _lookup_transform(self, target_frame, source_frame, stamp):
        """Non-blocking TF lookup. Prefer latest, then the image stamp if requested."""
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException as latest_exc:
            if not (self.prefer_image_timestamp and stamp is not None):
                raise latest_exc
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.0),
            )

    def _cache_camera_taskboard_poses(self, stamp):
        """Cache camera<->taskboard poses for this frame. Return True on success."""
        try:
            tf_cam_to_tb = self._lookup_transform(
                self.taskboard_frame, self.camera_frame, stamp
            )
        except Exception:
            self._cam_pos_tb = None
            self._cam_rot_tb = None
            self._tb_pos_cam = None
            self._tb_rot_cam = None
            return False

        t = tf_cam_to_tb.transform.translation
        q = tf_cam_to_tb.transform.rotation
        self._cam_pos_tb = np.array([t.x, t.y, t.z], dtype=np.float64)
        self._cam_rot_tb = self.quaternion_to_rotation_matrix(
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        )
        self._tb_rot_cam = self._cam_rot_tb.T
        self._tb_pos_cam = -self._tb_rot_cam @ self._cam_pos_tb
        return True

    def _pixel_to_camera_ray(self, pixel_x, pixel_y):
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        if self.dist_coeffs is not None and len(self.dist_coeffs) > 0 and np.any(self.dist_coeffs != 0):
            pixel_distorted = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            pixel_undistorted = cv2.undistortPoints(
                pixel_distorted,
                self.camera_matrix,
                self.dist_coeffs,
                P=self.camera_matrix,
            )
            pixel_x = float(pixel_undistorted[0, 0, 0])
            pixel_y = float(pixel_undistorted[0, 0, 1])
        ray_camera = np.array(
            [(pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0],
            dtype=np.float64,
        )
        return ray_camera / np.linalg.norm(ray_camera)

    def _intersect_pixel_with_plane(self, pixel_x, pixel_y, plane_z):
        if not self.camera_info_received:
            return None, 'camera_info not yet received'
        if self._cam_pos_tb is None or self._cam_rot_tb is None:
            return None, 'TF lookup failed'
        ray_camera = self._pixel_to_camera_ray(float(pixel_x), float(pixel_y))
        ray_taskboard = self._cam_rot_tb @ ray_camera
        if abs(ray_taskboard[2]) < 1e-6:
            return None, 'ray parallel to zone plane'
        t_intersect = (plane_z - self._cam_pos_tb[2]) / ray_taskboard[2]
        if t_intersect < 0:
            return None, 'intersection behind camera'
        return self._cam_pos_tb + t_intersect * ray_taskboard, None

    def _is_stale(self, received_time, last_warn_attr, label):
        if self.port_pixel_max_age_sec <= 0.0 or received_time is None:
            return False
        age_sec = (self.get_clock().now() - received_time).nanoseconds / 1e9
        if age_sec <= self.port_pixel_max_age_sec:
            return False
        now = self.get_clock().now()
        last_warn = getattr(self, last_warn_attr)
        if last_warn is None or (now - last_warn).nanoseconds > int(self.stale_warn_interval_sec * 1e9):
            self.get_logger().warn(
                f'Skipping stale {label} (age={age_sec:.3f}s > {self.port_pixel_max_age_sec:.3f}s).'
            )
            setattr(self, last_warn_attr, now)
        return True

    def _build_nic_port_targets(self):
        targets = []
        port_msg = self.latest_nic_port_pixels_msg
        if port_msg is None:
            return targets
        if self._is_stale(
            self.latest_nic_port_pixels_received_time,
            '_last_stale_port_warn_time',
            'NIC port pixels',
        ):
            return targets

        x_min, x_max, _, _ = self._zone_bounds(NIC_ZONE)
        grouped_by_rail = {i: [] for i in range(self.rail_count)}
        for pose in port_msg.poses:
            u = pose.position.x
            v = pose.position.y
            intersection, _ = self._intersect_pixel_with_plane(
                u, v, self.nic_zone_offset_m
            )
            if intersection is None or not self._point_in_zone(intersection, NIC_ZONE):
                continue
            rail_idx = self._zone_rail_index(intersection[1], NIC_ZONE)
            classified_port_idx = self._infer_port_index_from_nic_detection(u, v)
            grouped_by_rail[rail_idx].append((intersection, classified_port_idx))

        for rail_idx in range(self.rail_count):
            rail_points = sorted(grouped_by_rail[rail_idx], key=lambda item: item[0][0])
            if len(rail_points) > 2:
                now = self.get_clock().now()
                if (
                    self._last_overflow_port_warn_time is None
                    or (now - self._last_overflow_port_warn_time).nanoseconds
                    > int(self.rail_overflow_warn_interval_sec * 1e9)
                ):
                    self.get_logger().warn(
                        f'NIC rail {rail_idx} has {len(rail_points)} detections; '
                        'taking first two by x.'
                    )
                    self._last_overflow_port_warn_time = now
                rail_points = rail_points[:2]
            labeled_points = self._label_rail_points(
                rail_points=rail_points, x_side=x_min, x_middle=x_max
            )
            for port_idx, point in labeled_points:
                targets.append({'class_id': f'NIC_CARD_{rail_idx}_{port_idx}', 'point': point})
        return targets

    def _build_sc_port_targets(self):
        targets = []
        msg = self.latest_sc_detections_msg
        if msg is None:
            return targets
        if self._is_stale(
            self.latest_sc_detections_received_time,
            '_last_stale_sc_warn_time',
            'SC detections',
        ):
            return targets

        grouped_by_rail = {i: [] for i in range(self.sc_active_rail_count)}
        for det in msg.detections:
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)
            intersection, _ = self._intersect_pixel_with_plane(
                u, v, self.sc_zone_offset_m
            )
            if intersection is None or not self._point_in_zone(intersection, SC_ZONE):
                continue
            rail_idx = self._zone_rail_index(intersection[1], SC_ZONE)
            if rail_idx in grouped_by_rail:
                grouped_by_rail[rail_idx].append(intersection)

        for rail_idx in range(self.sc_active_rail_count):
            rail_points = grouped_by_rail[rail_idx]
            if not rail_points:
                continue
            point = np.mean(np.asarray(rail_points, dtype=np.float64), axis=0)
            targets.append({'class_id': f'SC_PORT_{rail_idx}', 'point': point})
        return targets

    def _infer_port_index_from_nic_detection(self, u_px, v_px):
        msg = self.latest_nic_detections_msg
        if msg is None:
            return None
        best_dist_sq = None
        best_class_id = None
        for det in msg.detections:
            if not det.results:
                continue
            du = float(det.bbox.center.position.x) - float(u_px)
            dv = float(det.bbox.center.position.y) - float(v_px)
            dist_sq = du * du + dv * dv
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_class_id = det.results[0].hypothesis.class_id
        if best_dist_sq is None:
            return None
        max_sq = self.nic_detection_association_max_px * self.nic_detection_association_max_px
        if best_dist_sq > max_sq:
            return None
        return self._class_id_to_port_index(best_class_id)

    @staticmethod
    def _class_id_to_port_index(class_id):
        token = str(class_id).strip().lower()
        if token in ('0', '1'):
            return int(token)
        if 'port_0' in token or token.endswith('port0'):
            return 0
        if 'port_1' in token or token.endswith('port1'):
            return 1
        return None

    @staticmethod
    def _label_rail_points(rail_points, x_side, x_middle):
        """Return list[(port_idx, point)] for one NIC rail.

        Port 1 is closer to the outer side, port 0 is closer to the board middle.
        """
        if not rail_points:
            return []

        labeled = {}
        unlabeled = []
        for point, classified_idx in rail_points:
            if classified_idx in (0, 1) and classified_idx not in labeled:
                labeled[classified_idx] = point
            else:
                unlabeled.append(point)

        if 0 not in labeled or 1 not in labeled:
            unlabeled_sorted = sorted(unlabeled, key=lambda p: p[0])
            if 0 not in labeled and 1 not in labeled:
                if len(unlabeled_sorted) >= 2:
                    labeled[1] = unlabeled_sorted[0]
                    labeled[0] = unlabeled_sorted[-1]
                else:
                    only = unlabeled_sorted[0] if unlabeled_sorted else rail_points[0][0]
                    dist_side = abs(float(only[0]) - float(x_side))
                    dist_mid = abs(float(only[0]) - float(x_middle))
                    labeled[1 if dist_side <= dist_mid else 0] = only
            else:
                missing_idx = 0 if 0 not in labeled else 1
                if unlabeled_sorted:
                    if missing_idx == 1:
                        labeled[1] = unlabeled_sorted[0]
                    else:
                        labeled[0] = unlabeled_sorted[-1]

        out = []
        if 0 in labeled:
            out.append((0, labeled[0]))
        if 1 in labeled:
            out.append((1, labeled[1]))
        return out

    def _publish_port_outputs(self, stamp):
        nic_targets = self._build_nic_port_targets()
        sc_targets = self._build_sc_port_targets()
        if self.publish_port_tfs:
            transforms = self._port_transforms_from_detections(
                nic_targets, stamp, self.nic_port_tf_prefix, parse_nic=True
            )
            transforms.extend(
                self._port_transforms_from_detections(
                    sc_targets, stamp, self.sc_port_tf_prefix, parse_nic=False
                )
            )
            if transforms:
                self.tf_broadcaster.sendTransform(transforms)
        if self.publish_port_entrance_tfs:
            self._update_port_entrance_latches(
                nic_targets, self.nic_port_entrance_tf_prefix, parse_nic=True
            )
            self._update_port_entrance_latches(
                sc_targets, self.sc_port_entrance_tf_prefix, parse_nic=False
            )
            entrance_transforms = self._latched_port_entrance_transforms(stamp)
            if entrance_transforms:
                self.tf_broadcaster.sendTransform(entrance_transforms)
        return nic_targets, sc_targets

    @staticmethod
    def _rail_port_from_class_id(class_id, parse_nic):
        parts = str(class_id).strip().split('_')
        if parse_nic:
            if len(parts) < 4 or parts[0].upper() != 'NIC' or parts[1].upper() != 'CARD':
                return None
            try:
                return int(parts[-2]), int(parts[-1])
            except ValueError:
                return None
        if len(parts) < 3 or parts[0].upper() != 'SC' or parts[1].upper() != 'PORT':
            return None
        try:
            return int(parts[-1]), 0
        except ValueError:
            return None

    def _port_transforms_from_detections(self, targets, stamp, prefix, parse_nic):
        """Build taskboard_detected -> port child TFs for the current frame."""
        transforms = []
        for target in targets:
            rp = self._rail_port_from_class_id(target['class_id'], parse_nic)
            if rp is None:
                continue
            rail_idx, port_idx = rp
            child_frame_id = (
                f'{prefix}_r{rail_idx}_p{port_idx}' if parse_nic else f'{prefix}_r{rail_idx}'
            )
            point = target['point']
            transforms.append(self._identity_rotation_transform(child_frame_id, point, stamp))
        return transforms

    def _identity_rotation_transform(self, child_frame_id, position, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.taskboard_frame
        t.child_frame_id = child_frame_id
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.w = 1.0
        return t

    def _latched_port_entrance_transforms(self, stamp):
        return [
            self._identity_rotation_transform(child_frame_id, position, stamp)
            for child_frame_id, position in self._latched_port_entrance_frames.items()
        ]

    def _update_port_entrance_latches(self, targets, prefix, parse_nic):
        prefix = prefix or ('nic_port_entrance' if parse_nic else 'sc_port_entrance')
        seen_this_frame = set()
        for target in targets:
            class_id = target['class_id']
            rp = self._rail_port_from_class_id(class_id, parse_nic)
            if rp is None:
                continue
            rail_idx, port_idx = rp
            child_frame_id = (
                f'{prefix}_r{rail_idx}_p{port_idx}' if parse_nic else f'{prefix}_r{rail_idx}'
            )
            seen_this_frame.add(child_frame_id)
            if child_frame_id in self._latched_port_entrance_frames:
                continue

            point = target['point']
            position = np.array(
                [float(point[0]), float(point[1]), float(point[2])],
                dtype=np.float64,
            )
            prev_position = self._port_entrance_candidate_positions.get(child_frame_id)
            if prev_position is None:
                hits = 1
                self._port_entrance_candidate_positions[child_frame_id] = position
            else:
                position_shift = float(np.linalg.norm(position - prev_position))
                if position_shift <= self.port_entrance_position_epsilon_m:
                    hits = self._port_entrance_candidate_hits.get(child_frame_id, 0) + 1
                    self._port_entrance_candidate_positions[child_frame_id] = position
                else:
                    hits = 1
                    self._port_entrance_candidate_positions[child_frame_id] = position
            self._port_entrance_candidate_hits[child_frame_id] = hits
            if hits < self.port_entrance_min_consecutive_hits:
                continue

            self._latched_port_entrance_frames[child_frame_id] = position
            self._port_entrance_candidate_positions.pop(child_frame_id, None)
            self._port_entrance_candidate_hits.pop(child_frame_id, None)
            self.get_logger().info(
                f'Latched port entrance TF "{child_frame_id}" after '
                f'{hits} consecutive stable detections.'
            )

        stale_candidates = [
            child
            for child in list(self._port_entrance_candidate_hits)
            if child.startswith(prefix) and child not in seen_this_frame
        ]
        for child in stale_candidates:
            self._port_entrance_candidate_hits.pop(child, None)
            self._port_entrance_candidate_positions.pop(child, None)

    def transform_taskboard_points_to_camera(self, points_taskboard, stamp=None):
        if self._tb_rot_cam is None or self._tb_pos_cam is None:
            raise TransformException('camera/taskboard pose not cached')
        return (self._tb_rot_cam @ points_taskboard.T).T + self._tb_pos_cam

    def project_points(self, points_camera):
        if np.any(points_camera[:, 2] <= self.min_projected_depth_m):
            return None
        zero_rvec = np.zeros((3, 1), dtype=np.float64)
        zero_tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            points_camera.astype(np.float64),
            zero_rvec,
            zero_tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return image_points.reshape(-1, 2)

    def _draw_zone_overlay(self, cv_image, stamp, zone_number, plane_z, color, label):
        try:
            corners_camera = self.transform_taskboard_points_to_camera(
                self._zone_corners(zone_number, plane_z), stamp
            )
        except Exception:
            return
        corners_px = self.project_points(corners_camera)
        if corners_px is None:
            return
        polygon = np.round(corners_px).astype(np.int32).reshape(-1, 1, 2)
        overlay = cv_image.copy()
        cv2.fillPoly(overlay, [polygon], color=color)
        cv2.addWeighted(
            overlay, self.overlay_alpha, cv_image, max(0.0, 1.0 - self.overlay_alpha), 0.0, cv_image
        )
        cv2.polylines(cv_image, [polygon], True, color=color, thickness=3)

        if self.rail_count > 1:
            for segment in self._zone_rail_boundaries(zone_number, plane_z):
                try:
                    segment_camera = self.transform_taskboard_points_to_camera(segment, stamp)
                except Exception:
                    break
                segment_px = self.project_points(segment_camera)
                if segment_px is None:
                    continue
                p0 = tuple(np.round(segment_px[0]).astype(np.int32))
                p1 = tuple(np.round(segment_px[1]).astype(np.int32))
                cv2.line(cv_image, p0, p1, color=RAIL_COLOR, thickness=2)

        centroid = np.round(np.mean(corners_px, axis=0)).astype(np.int32)
        cv2.putText(
            cv_image,
            f'{label} ({plane_z:+.2f}m)',
            (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_target_markers(self, cv_image, stamp, targets, color):
        for target in targets:
            point = np.asarray(target['point'], dtype=np.float32).reshape(1, 3)
            try:
                point_camera = self.transform_taskboard_points_to_camera(point, stamp)
            except Exception:
                continue
            point_px = self.project_points(point_camera)
            if point_px is None:
                continue
            px, py = int(round(point_px[0][0])), int(round(point_px[0][1]))
            cv2.drawMarker(cv_image, (px, py), color, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                cv_image,
                target['class_id'],
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    def image_callback(self, msg):
        nic_targets = None
        sc_targets = None
        tf_ready = False
        if self.camera_info_received:
            tf_ready = self._cache_camera_taskboard_poses(msg.header.stamp)
            if tf_ready:
                self._activate_if_tf_ready()
                nic_targets, sc_targets = self._publish_port_outputs(msg.header.stamp)
        else:
            self._throttled_info(
                f'Waiting for camera_info on {self.input_camera_info_topic}'
            )

        if not self.publish_debug_visualization or self.image_pub is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        if not self.camera_info_received:
            cv2.putText(
                cv_image,
                'Waiting for camera_info...',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
            self._publish_image(msg, cv_image)
            return

        if not tf_ready:
            cv2.putText(
                cv_image,
                f'Waiting for TF: {self.taskboard_frame}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
            self._publish_image(msg, cv_image)
            return

        self._draw_zone_overlay(
            cv_image,
            msg.header.stamp,
            NIC_ZONE,
            self.nic_zone_offset_m,
            NIC_ZONE_COLOR,
            'Zone 1 NIC',
        )
        self._draw_zone_overlay(
            cv_image,
            msg.header.stamp,
            SC_ZONE,
            self.sc_zone_offset_m,
            SC_ZONE_COLOR,
            'Zone 2 SC',
        )
        if nic_targets is not None:
            self._draw_target_markers(cv_image, msg.header.stamp, nic_targets, NIC_POINT_COLOR)
        if sc_targets is not None:
            self._draw_target_markers(cv_image, msg.header.stamp, sc_targets, SC_POINT_COLOR)

        nic_n = 0 if nic_targets is None else len(nic_targets)
        sc_n = 0 if sc_targets is None else len(sc_targets)
        cv2.putText(
            cv_image,
            f'NIC targets={nic_n}  SC targets={sc_n}',
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )
        self._publish_image(msg, cv_image)

    def _publish_image(self, image_msg, cv_image):
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.image_pub.publish(out_msg)


def main(args=None):
    parser = ArgumentParser(add_help=True)
    _, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = ZoneProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt. Exiting...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
