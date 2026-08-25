#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


class TaskboardTfFusion(Node):
    """Fuse per-camera taskboard TFs into one base_link -> taskboard_detected pose."""

    def __init__(self):
        super().__init__('taskboard_tf_fusion')

        self.camera_names = [
            str(c).strip()
            for c in self.declare_parameter('camera_names', ['center', 'left', 'right']).value
            if str(c).strip()
        ]
        self.camera_taskboard_frame_prefix = str(
            self.declare_parameter('camera_taskboard_frame_prefix', 'taskboard_').value
        )
        self.parent_frame = str(self.declare_parameter('parent_frame', 'base_link').value)
        self.taskboard_frame = str(
            self.declare_parameter('taskboard_frame', 'taskboard_detected').value
        )
        self.min_cameras_for_lock = int(self.declare_parameter('min_cameras_for_lock', 2).value)
        self.continue_detecting = bool(self.declare_parameter('continue_detecting', False).value)
        self.tf_lookup_timeout_sec = float(self.declare_parameter('tf_lookup_timeout_sec', 0.1).value)
        self.tf_max_age_sec = float(self.declare_parameter('tf_max_age_sec', 0.5).value)
        self.update_period_sec = float(self.declare_parameter('update_period_sec', 0.05).value)
        self.taskboard_projection_position_threshold_m = float(
            self.declare_parameter('taskboard_projection_position_threshold_m', 0.05).value
        )
        self.taskboard_projection_rotation_similarity_threshold = float(
            self.declare_parameter('taskboard_projection_rotation_similarity_threshold', 0.99).value
        )
        self.taskboard_face_up_min_dot = float(
            self.declare_parameter('taskboard_face_up_min_dot', 0.85).value
        )
        self.enable_taskboard_pose_flattening = bool(
            self.declare_parameter('enable_taskboard_pose_flattening', True).value
        )
        self.taskboard_flatten_face_up = bool(
            self.declare_parameter('taskboard_flatten_face_up', True).value
        )
        self.publish_debug_transforms = bool(
            self.declare_parameter('publish_debug_transforms', False).value
        )
        self.detection_state_topic = str(
            self.declare_parameter('detection_state_topic', 'taskboard_detection_state').value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.detection_state_pub = self.create_publisher(Bool, self.detection_state_topic, 10)

        self.taskboard_detected = False
        self._locked_transform = None

        self.create_timer(max(0.01, self.update_period_sec), self._timer_callback)
        self._publish_detection_state(False)

        camera_frames = [self._camera_taskboard_frame(name) for name in self.camera_names]
        self.get_logger().info('Taskboard TF fusion node started.')
        self.get_logger().info(
            f'Waiting for 2-{len(self.camera_names)} corresponding TFs among {camera_frames}'
        )
        self.get_logger().info(
            f'On match, publishing {self.parent_frame} -> {self.taskboard_frame}'
        )
        self.get_logger().info(
            f'continue_detecting={self.continue_detecting} '
            f'(false = lock first match and keep TF alive, true = keep updating)'
        )
        self.get_logger().info(
            f'Taskboard face-up sanity gate: normal·{self.parent_frame}_up >= '
            f'{self.taskboard_face_up_min_dot:.2f}'
        )
        self.get_logger().info(
            f'Taskboard pose flattening: enabled={self.enable_taskboard_pose_flattening}, '
            f'face_up={self.taskboard_flatten_face_up}'
        )

    def _camera_taskboard_frame(self, camera_name):
        return f'{self.camera_taskboard_frame_prefix}{camera_name}'

    def _publish_detection_state(self, detected):
        msg = Bool()
        msg.data = bool(detected)
        self.detection_state_pub.publish(msg)

    def _timer_callback(self):
        if self._locked_transform is not None:
            self._republish_locked_transform()
            if not self.continue_detecting:
                return

        fused = self._try_fuse_camera_transforms()
        if fused is None:
            return

        was_detected = self.taskboard_detected
        self._locked_transform = fused
        self.taskboard_detected = True
        self._republish_locked_transform()
        self._publish_detection_state(True)

        if not was_detected:
            t = fused.transform.translation
            self.get_logger().info(
                f'Taskboard detected! Publishing {self.parent_frame}->{self.taskboard_frame}. '
                f'Position: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})m'
            )
            if self.continue_detecting:
                self.get_logger().info(
                    'continue_detecting is true: pose will keep updating from matching camera TFs.'
                )
            else:
                self.get_logger().info(
                    'Locked first match. Detection stopped; node stays alive to keep '
                    f'{self.taskboard_frame} available to downstream nodes.'
                )

    def _republish_locked_transform(self):
        if self._locked_transform is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._locked_transform.header.frame_id
        t.child_frame_id = self._locked_transform.child_frame_id
        t.transform = self._locked_transform.transform
        self.tf_broadcaster.sendTransform(t)

    def _try_fuse_camera_transforms(self):
        valid_transforms = []
        valid_cameras = []
        timeout = Duration(seconds=max(0.0, self.tf_lookup_timeout_sec))
        now = self.get_clock().now()

        for camera_name in self.camera_names:
            child_frame = self._camera_taskboard_frame(camera_name)
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.parent_frame,
                    child_frame,
                    Time(),
                    timeout=timeout,
                )
            except TransformException as exc:
                self.get_logger().debug(
                    f'No {self.parent_frame}->{child_frame} transform: {exc}'
                )
                continue

            if not self._transform_is_fresh(tf_msg, now):
                continue

            rot, trans = self._transform_to_rot_trans(tf_msg)
            valid_transforms.append((rot, trans))
            valid_cameras.append(camera_name)

        if len(valid_transforms) < self.min_cameras_for_lock:
            self.get_logger().debug(
                f'Only {len(valid_transforms)} camera TF(s) in {self.parent_frame}. '
                f'Need at least {self.min_cameras_for_lock}.',
            )
            return None

        if self.publish_debug_transforms:
            self._publish_debug_transforms(valid_transforms, valid_cameras)

        projections_similar, consensus_indices = self.verify_taskboard_projections(
            valid_transforms, valid_cameras
        )
        if not projections_similar or len(consensus_indices) < self.min_cameras_for_lock:
            self.get_logger().warn(
                f'Taskboard TFs too different across cameras {valid_cameras}. Skipping detection.',
                throttle_duration_sec=2.0,
            )
            return None

        consensus_transforms = [valid_transforms[i] for i in consensus_indices]
        consensus_cameras = [valid_cameras[i] for i in consensus_indices]

        quats = np.array(
            [self.rotation_matrix_to_quaternion(rot) for rot, _ in consensus_transforms]
        )
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] *= -1

        avg_quat = quats.mean(axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        avg_rot = self.quaternion_to_rotation_matrix(avg_quat)
        board_normal = avg_rot[:, 2]
        up_alignment = float(np.dot(board_normal, np.array([0.0, 0.0, 1.0])))
        if up_alignment < self.taskboard_face_up_min_dot:
            self.get_logger().warn(
                'Taskboard orientation sanity check failed: '
                f'normal·{self.parent_frame}_up={up_alignment:.3f} < '
                f'{self.taskboard_face_up_min_dot:.3f}. Rejecting match.',
                throttle_duration_sec=2.0,
            )
            return None

        if self.enable_taskboard_pose_flattening:
            avg_rot = self.flatten_taskboard_rotation_to_table(avg_rot)
            avg_quat = self.rotation_matrix_to_quaternion(avg_rot)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)

        avg_trans = np.array([trans for _, trans in consensus_transforms]).mean(axis=0)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.taskboard_frame
        t.transform.translation.x = float(avg_trans[0])
        t.transform.translation.y = float(avg_trans[1])
        t.transform.translation.z = float(avg_trans[2])
        t.transform.rotation.x = float(avg_quat[0])
        t.transform.rotation.y = float(avg_quat[1])
        t.transform.rotation.z = float(avg_quat[2])
        t.transform.rotation.w = float(avg_quat[3])

        self.get_logger().info(
            f'Fused taskboard pose from cameras {consensus_cameras}: '
            f'pos=({avg_trans[0]:.3f}, {avg_trans[1]:.3f}, {avg_trans[2]:.3f})m, '
            f'normal·up={up_alignment:.3f}',
            throttle_duration_sec=1.0,
        )
        return t

    def _transform_is_fresh(self, tf_msg, now):
        stamp = tf_msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return True
        age_sec = abs((now - Time.from_msg(stamp)).nanoseconds) / 1e9
        if age_sec > self.tf_max_age_sec:
            self.get_logger().debug(
                f'Stale {tf_msg.child_frame_id} TF age={age_sec:.3f}s '
                f'(max {self.tf_max_age_sec:.3f}s)'
            )
            return False
        return True

    def _transform_to_rot_trans(self, tf_msg):
        quat = np.array(
            [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ]
        )
        trans = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ]
        )
        return self.quaternion_to_rotation_matrix(quat), trans

    def _publish_debug_transforms(self, valid_transforms, camera_names):
        stamp = self.get_clock().now().to_msg()
        debug_transforms = []
        for (rot, trans), camera_name in zip(valid_transforms, camera_names):
            quat = self.rotation_matrix_to_quaternion(rot)
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.parent_frame
            t.child_frame_id = f'{self.taskboard_frame}_debug_{camera_name}'
            t.transform.translation.x = float(trans[0])
            t.transform.translation.y = float(trans[1])
            t.transform.translation.z = float(trans[2])
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            debug_transforms.append(t)
        if debug_transforms:
            self.tf_broadcaster.sendTransform(debug_transforms)

    def verify_taskboard_projections(self, valid_transforms, camera_names):
        """Keep a consensus subset of camera TFs that agree in parent_frame."""
        if len(valid_transforms) < 2:
            return True, list(range(len(valid_transforms)))

        positions = np.array([trans for _, trans in valid_transforms])
        rotations = [rot for rot, _ in valid_transforms]
        pairwise_distances = []
        pairwise_sims = []
        pairwise_indices = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pairwise_distances.append(float(np.linalg.norm(positions[i] - positions[j])))
                pairwise_indices.append((i, j))

        max_distance = max(pairwise_distances) if pairwise_distances else 0.0
        position_threshold = self.taskboard_projection_position_threshold_m
        rotation_threshold = self.taskboard_projection_rotation_similarity_threshold

        quats = np.array([self.rotation_matrix_to_quaternion(rot) for rot in rotations])
        for i, j in pairwise_indices:
            pairwise_sims.append(abs(float(np.dot(quats[i], quats[j]))))

        min_similarity = min(pairwise_sims) if pairwise_sims else 1.0
        all_valid = (max_distance <= position_threshold) and (min_similarity >= rotation_threshold)

        passing_pairs = [
            (i, j)
            for (i, j), dist, sim in zip(pairwise_indices, pairwise_distances, pairwise_sims)
            if dist <= position_threshold and sim >= rotation_threshold
        ]

        if all_valid:
            self.get_logger().info(
                f'Projection verification passed for {camera_names}: '
                f'position_diff={max_distance:.4f}m, rotation_sim={min_similarity:.4f}',
                throttle_duration_sec=1.0,
            )
            return True, list(range(len(valid_transforms)))

        if passing_pairs:
            best_i, best_j = min(
                passing_pairs, key=lambda ij: np.linalg.norm(positions[ij[0]] - positions[ij[1]])
            )
            consensus = {best_i, best_j}
            for k in range(len(valid_transforms)):
                if k in consensus:
                    continue
                consistent = True
                for c in consensus:
                    dist = float(np.linalg.norm(positions[k] - positions[c]))
                    sim = float(abs(np.dot(quats[k], quats[c])))
                    if dist > position_threshold or sim < rotation_threshold:
                        consistent = False
                        break
                if consistent:
                    consensus.add(k)

            consensus_indices = sorted(consensus)
            consensus_cameras = [camera_names[i] for i in consensus_indices]
            self.get_logger().warn(
                f'Verification accepted consensus subset {consensus_cameras} out of {camera_names}; '
                'outlier cameras ignored.',
                throttle_duration_sec=2.0,
            )
            return True, consensus_indices

        pos_status = 'ok' if max_distance <= position_threshold else 'fail'
        rot_status = 'ok' if min_similarity >= rotation_threshold else 'fail'
        self.get_logger().warn(
            f'Projection verification failed for {camera_names}: '
            f'{pos_status} position_diff={max_distance:.4f}m (threshold={position_threshold}m) | '
            f'{rot_status} rotation_sim={min_similarity:.4f} (threshold={rotation_threshold})',
            throttle_duration_sec=2.0,
        )
        return False, []

    def flatten_taskboard_rotation_to_table(self, taskboard_rot):
        """Force taskboard normal to parent +/-Z while preserving in-plane yaw."""
        x_axis = np.array(taskboard_rot[:, 0], dtype=np.float64)
        x_proj = x_axis.copy()
        x_proj[2] = 0.0
        if np.linalg.norm(x_proj) < 1e-6:
            y_axis = np.array(taskboard_rot[:, 1], dtype=np.float64)
            y_proj = y_axis.copy()
            y_proj[2] = 0.0
            if np.linalg.norm(y_proj) < 1e-6:
                x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                y_proj = y_proj / np.linalg.norm(y_proj)
                x_proj = np.array([y_proj[1], -y_proj[0], 0.0], dtype=np.float64)
        x_flat = x_proj / max(1e-9, np.linalg.norm(x_proj))

        z_sign = 1.0 if self.taskboard_flatten_face_up else -1.0
        z_flat = np.array([0.0, 0.0, z_sign], dtype=np.float64)
        y_flat = np.cross(z_flat, x_flat)
        y_flat = y_flat / max(1e-9, np.linalg.norm(y_flat))
        x_flat = np.cross(y_flat, z_flat)
        x_flat = x_flat / max(1e-9, np.linalg.norm(x_flat))
        return np.column_stack((x_flat, y_flat, z_flat))

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    def quaternion_to_rotation_matrix(self, quat):
        x, y, z, w = quat
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )


def main(args=None):
    parser = ArgumentParser(add_help=True)
    _, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = TaskboardTfFusion()
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
