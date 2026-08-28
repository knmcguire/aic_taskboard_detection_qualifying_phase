#!/usr/bin/env python3

"""Translate 2D SC/NIC port detections into 3D table-frame poses.

Inputs:
  - CameraInfo (intrinsics + optical frame)
  - Detection2DArray from component detection (SC and NIC)
  - TF ``table_frame`` <-> camera optical frame

Each detection center is cast as a camera ray and intersected with a
horizontal plane in ``table_frame``:
  - SC ports:  3 cm  (``sc_plane_offset_m``)
  - NIC ports: 15 cm (``nic_plane_offset_m``)

One child TF is published per detection (no count limit). Names match
zone_projection, with a couple index instead of a rail index:
  ``{nic_port_tf_prefix}_r{couple}_p{port}``
  ``{sc_port_tf_prefix}_r{couple}``
"""

import math
from argparse import ArgumentParser

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


class SimplePort3D(Node):
    """Ray-plane lift of 2D port detections; publishes one TF per hit."""

    def __init__(self):
        super().__init__('simple_port_3d')

        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        default_camera_info = f'/{self.camera_name}_camera/camera_info'
        default_camera_frame = f'{self.camera_name}_camera/optical'

        self.camera_info_topic = str(
            self.declare_parameter('camera_info_topic', default_camera_info).value
        )
        self.nic_detections_topic = str(
            self.declare_parameter(
                'nic_detections_topic', '/nic_port_detection/detections'
            ).value
        )
        self.sc_detections_topic = str(
            self.declare_parameter(
                'sc_detections_topic', '/sc_port_detection/detections'
            ).value
        )
        self.nic_port_targets_topic = str(
            self.declare_parameter(
                'nic_port_targets_topic', '/simple_port_3d/nic_port_targets'
            ).value
        )
        self.sc_port_targets_topic = str(
            self.declare_parameter(
                'sc_port_targets_topic', '/simple_port_3d/sc_port_targets'
            ).value
        )
        self.table_frame = str(self.declare_parameter('table_frame', 'tabletop').value).strip()
        self.camera_frame = str(
            self.declare_parameter('camera_frame_fallback', default_camera_frame).value
        ).strip()
        self.use_camera_info_frame = bool(
            self.declare_parameter('use_camera_info_frame', True).value
        )
        self.nic_plane_offset_m = float(self.declare_parameter('nic_plane_offset_m', 0.15).value)
        self.sc_plane_offset_m = float(self.declare_parameter('sc_plane_offset_m', 0.03).value)
        self.nic_port_tf_prefix = str(
            self.declare_parameter('nic_port_tf_prefix', 'nic_port').value
        ).strip() or 'nic_port'
        self.sc_port_tf_prefix = str(
            self.declare_parameter('sc_port_tf_prefix', 'sc_port').value
        ).strip() or 'sc_port'
        self.tf_lookup_timeout_sec = float(
            self.declare_parameter('tf_lookup_timeout_sec', 0.2).value
        )
        tf_buffer_duration_sec = max(
            1.0, float(self.declare_parameter('tf_buffer_duration_sec', 10.0).value)
        )
        self.ray_epsilon = float(self.declare_parameter('ray_epsilon', 1e-9).value)
        self.yaw_baseline_epsilon = float(
            self.declare_parameter('yaw_baseline_epsilon', 1e-6).value
        )
        self.target_bbox_size_m = float(self.declare_parameter('target_bbox_size_m', 0.001).value)
        self.status_period_sec = max(
            0.5, float(self.declare_parameter('status_period_sec', 2.0).value)
        )
        self.camera_info_received = False
        self.camera_matrix = None
        self._cam_pos_table = None
        self._cam_rot_table = None
        self._last_tf_error = None
        self._last_nic_count = 0
        self._last_sc_count = 0

        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, camera_info_qos
        )
        self.create_subscription(
            Detection2DArray,
            self.nic_detections_topic,
            self._on_nic_detections,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Detection2DArray,
            self.sc_detections_topic,
            self._on_sc_detections,
            qos_profile_sensor_data,
        )
        self.nic_port_targets_pub = self.create_publisher(
            Detection3DArray, self.nic_port_targets_topic, qos_profile_sensor_data
        )
        self.sc_port_targets_pub = self.create_publisher(
            Detection3DArray, self.sc_port_targets_topic, qos_profile_sensor_data
        )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=tf_buffer_duration_sec))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(self.status_period_sec, self._log_status)

        self.get_logger().info(
            'SimplePort3D ready. '
            f'camera_info={self.camera_info_topic} table_frame={self.table_frame} '
            f'nic_z={self.nic_plane_offset_m:.3f}m sc_z={self.sc_plane_offset_m:.3f}m'
        )
        self.get_logger().info(f'NIC detections: {self.nic_detections_topic}')
        self.get_logger().info(f'SC detections:  {self.sc_detections_topic}')
        self.get_logger().info(
            f'TFs: {self.table_frame} -> {self.nic_port_tf_prefix}_r*_p* / '
            f'{self.sc_port_tf_prefix}_r* (couple index, not rail)'
        )

    def _on_camera_info(self, msg):
        self.camera_info_received = True
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if self.use_camera_info_frame and msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def _log_status(self):
        if not self.camera_info_received:
            self.get_logger().warn(
                f'Waiting for camera_info on {self.camera_info_topic}'
            )
            return
        if not self._refresh_camera_table_pose():
            error = self._last_tf_error or 'unknown TF error'
            self.get_logger().warn(
                f'Cannot lift detections to 3D: no TF {self.camera_frame} -> '
                f'{self.table_frame} ({error})'
            )
            return
        self.get_logger().info(
            f'3D lift ok: nic_tfs={self._last_nic_count} sc_tfs={self._last_sc_count} '
            f'plane={self.table_frame}'
        )

    @staticmethod
    def _quat_to_rot_matrix(x, y, z, w):
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_to_quat(yaw_rad):
        half = 0.5 * yaw_rad
        return math.sin(half), math.cos(half)

    @staticmethod
    def _nic_port_kind(label):
        text = str(label).strip().lower()
        if text in {'0', 'port_0', 'nic_port_0'} or text.endswith('port_0'):
            return 0
        if text in {'1', 'port_1', 'nic_port_1'} or text.endswith('port_1'):
            return 1
        return None

    def _yaw_from_port0_to_port1(self, port_0_xyz, port_1_xyz):
        line_xy = (port_1_xyz - port_0_xyz)[:2]
        if float(np.linalg.norm(line_xy)) < self.yaw_baseline_epsilon:
            return None
        yaw_deg = math.degrees(math.atan2(float(line_xy[1]), float(line_xy[0]))) + 180.0
        yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
        return math.radians(yaw_deg)

    def _nic_child_frame(self, couple_idx, port_idx):
        return f'{self.nic_port_tf_prefix}_r{couple_idx}_p{port_idx}'

    def _sc_child_frame(self, couple_idx):
        return f'{self.sc_port_tf_prefix}_r{couple_idx}'

    def _pair_nic_hits(self, hits):
        """Pair neighboring NIC ports and name them like zone_projection.

        zone_projection uses ``nic_port_r{rail}_p{port}``. Here the first
        index is a couple number (nearest port_0/port_1 pair), not a rail.
        """
        count = len(hits)
        if count == 0:
            return []

        kinds = [self._nic_port_kind(label) for label, _score, _point, _yaw in hits]
        points = [point for _label, _score, point, _yaw in hits]
        yaws = [None] * count
        couple_of = [None] * count
        port_of = [
            kind if kind in (0, 1) else 0
            for kind in kinds
        ]

        idx0 = [i for i, kind in enumerate(kinds) if kind == 0]
        idx1 = [i for i, kind in enumerate(kinds) if kind == 1]
        used1 = set()
        pairs = []

        for i in idx0:
            best_j = None
            best_dist = None
            for j in idx1:
                if j in used1:
                    continue
                dist = float(np.linalg.norm(points[i][:2] - points[j][:2]))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is None:
                continue
            used1.add(best_j)
            yaw = self._yaw_from_port0_to_port1(points[i], points[best_j])
            yaws[i] = yaw
            yaws[best_j] = yaw
            centroid = 0.5 * (points[i][:2] + points[best_j][:2])
            pairs.append((float(centroid[0]), float(centroid[1]), [i, best_j]))

        unmatched = [i for i in range(count) if i not in {m for _x, _y, members in pairs for m in members}]
        for i in unmatched:
            best_j = None
            best_dist = None
            for j in range(count):
                if i == j:
                    continue
                dist = float(np.linalg.norm(points[i][:2] - points[j][:2]))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None and yaws[i] is None:
                kind_i = kinds[i]
                kind_j = kinds[best_j]
                if kind_i == 1 or kind_j == 0:
                    yaws[i] = self._yaw_from_port0_to_port1(points[best_j], points[i])
                else:
                    yaws[i] = self._yaw_from_port0_to_port1(points[i], points[best_j])
            centroid = points[i][:2]
            pairs.append((float(centroid[0]), float(centroid[1]), [i]))

        pairs.sort(key=lambda item: (item[0], item[1]))
        for couple_idx, (_x, _y, members) in enumerate(pairs):
            for i in members:
                couple_of[i] = couple_idx

        return [
            (
                label,
                score,
                point,
                yaws[i],
                self._nic_child_frame(couple_of[i], port_of[i]),
            )
            for i, (label, score, point, _yaw) in enumerate(hits)
        ]

    @staticmethod
    def _detection_label(det):
        if not det.results:
            return ''
        return str(det.results[0].hypothesis.class_id).strip()

    @staticmethod
    def _detection_score(det):
        if not det.results:
            return 0.0
        return float(det.results[0].hypothesis.score)

    def _refresh_camera_table_pose(self):
        try:
            tf_cam_to_table = self.tf_buffer.lookup_transform(
                self.table_frame,
                self.camera_frame,
                Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException as exc:
            self._cam_pos_table = None
            self._cam_rot_table = None
            self._last_tf_error = str(exc)
            return False
        t = tf_cam_to_table.transform.translation
        q = tf_cam_to_table.transform.rotation
        self._cam_pos_table = np.array([t.x, t.y, t.z], dtype=np.float64)
        self._cam_rot_table = self._quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        self._last_tf_error = None
        return True

    def _pixel_to_camera_ray(self, u, v):
        if self.camera_matrix is None:
            return None
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        if abs(fx) < self.ray_epsilon or abs(fy) < self.ray_epsilon:
            return None
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(ray))
        if norm < self.ray_epsilon:
            return None
        return ray / norm

    def _intersect_pixel_with_offset(self, u, v, plane_offset_m):
        if self._cam_pos_table is None or self._cam_rot_table is None:
            return None
        ray_cam = self._pixel_to_camera_ray(u, v)
        if ray_cam is None:
            return None
        ray_table = self._cam_rot_table @ ray_cam
        if abs(ray_table[2]) < self.ray_epsilon:
            return None
        scale = (plane_offset_m - self._cam_pos_table[2]) / ray_table[2]
        if scale <= 0.0:
            return None
        return self._cam_pos_table + scale * ray_table

    def _image_theta_to_table_yaw(self, u, v, image_theta_rad, plane_offset_m):
        p0 = self._intersect_pixel_with_offset(u, v, plane_offset_m)
        if p0 is None:
            return None
        probe_px = 20.0
        u2 = u + probe_px * math.cos(image_theta_rad)
        v2 = v + probe_px * math.sin(image_theta_rad)
        p1 = self._intersect_pixel_with_offset(u2, v2, plane_offset_m)
        if p1 is None:
            return None
        dxy = p1[:2] - p0[:2]
        if float(np.linalg.norm(dxy)) < self.yaw_baseline_epsilon:
            return None
        yaw = float(math.atan2(float(dxy[1]), float(dxy[0])))
        yaw += math.pi
        return float((yaw + math.pi) % (2.0 * math.pi) - math.pi)

    def _make_tf(self, stamp, child_frame_id, point, qz, qw):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.table_frame
        t.child_frame_id = child_frame_id
        t.transform.translation.x = float(point[0])
        t.transform.translation.y = float(point[1])
        t.transform.translation.z = float(point[2])
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        return t

    def _make_detection3d(self, stamp, class_id, point, qz, qw, score):
        det = Detection3D()
        det.header.stamp = stamp
        det.header.frame_id = self.table_frame
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_id
        hyp.hypothesis.score = float(score)
        det.results.append(hyp)
        det.bbox.center.position.x = float(point[0])
        det.bbox.center.position.y = float(point[1])
        det.bbox.center.position.z = float(point[2])
        det.bbox.center.orientation.z = qz
        det.bbox.center.orientation.w = qw
        det.bbox.size.x = self.target_bbox_size_m
        det.bbox.size.y = self.target_bbox_size_m
        det.bbox.size.z = self.target_bbox_size_m
        return det

    def _sorted_detections(self, msg):
        entries = []
        for det in msg.detections:
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)
            entries.append((u, v, det))
        entries.sort(key=lambda item: (item[0], item[1]))
        return entries

    def _publish_hits(self, stamp, hits, targets_pub):
        tf_msgs = []
        targets = Detection3DArray()
        targets.header.stamp = stamp
        targets.header.frame_id = self.table_frame
        for label, score, point, yaw, child in hits:
            qz, qw = (self._yaw_to_quat(yaw) if yaw is not None else (0.0, 1.0))
            class_id = f'{child}:{label}' if label else child
            tf_msgs.append(self._make_tf(stamp, child, point, qz, qw))
            targets.detections.append(
                self._make_detection3d(stamp, class_id, point, qz, qw, score)
            )
        if tf_msgs:
            self.tf_broadcaster.sendTransform(tf_msgs)
        if targets.detections:
            targets_pub.publish(targets)
        return len(tf_msgs)

    def _on_nic_detections(self, msg):
        if not self.camera_info_received or not self._refresh_camera_table_pose():
            self._last_nic_count = 0
            return

        hits = []
        for u, v, det in self._sorted_detections(msg):
            point = self._intersect_pixel_with_offset(u, v, self.nic_plane_offset_m)
            if point is None:
                continue
            hits.append(
                (self._detection_label(det), self._detection_score(det), point, None)
            )
        hits = self._pair_nic_hits(hits)
        self._last_nic_count = self._publish_hits(
            msg.header.stamp, hits, self.nic_port_targets_pub
        )

    def _on_sc_detections(self, msg):
        if not self.camera_info_received or not self._refresh_camera_table_pose():
            self._last_sc_count = 0
            return

        hits = []
        for u, v, det in self._sorted_detections(msg):
            point = self._intersect_pixel_with_offset(u, v, self.sc_plane_offset_m)
            if point is None:
                continue
            yaw = self._image_theta_to_table_yaw(
                u, v, float(det.bbox.center.theta), self.sc_plane_offset_m
            )
            hits.append(
                (self._detection_label(det), self._detection_score(det), point, yaw)
            )
        named = [
            (label, score, point, yaw, self._sc_child_frame(idx))
            for idx, (label, score, point, yaw) in enumerate(hits)
        ]
        self._last_sc_count = self._publish_hits(
            msg.header.stamp, named, self.sc_port_targets_pub
        )


def main(args=None):
    parser = ArgumentParser(add_help=True)
    _, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = SimplePort3D()
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
