#!/usr/bin/env python3

from argparse import ArgumentParser

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


class RansacHoughTaskboardDetection(Node):
    """Detect a taskboard rectangle from a Canny image and publish that camera's PnP pose."""

    def __init__(self):
        super().__init__('ransac_hough_taskboard_detection')

        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        self.canny_image_topic = str(
            self.declare_parameter(
                'canny_image_topic', f'/{self.camera_name}_camera/image_canny'
            ).value
        )
        self.camera_info_topic = str(
            self.declare_parameter(
                'camera_info_topic', f'/{self.camera_name}_camera/camera_info'
            ).value
        )
        self.debug_image_topic = str(
            self.declare_parameter(
                'debug_image_topic', f'/{self.camera_name}_camera/image_taskboard'
            ).value
        )
        self.color_logo_center_topic = str(
            self.declare_parameter(
                'color_logo_center_topic', f'/{self.camera_name}_camera/color_logo_center'
            ).value
        )
        self.camera_optical_frame = str(
            self.declare_parameter(
                'camera_optical_frame', f'{self.camera_name}_camera/optical'
            ).value
        )
        self.camera_taskboard_frame = str(
            self.declare_parameter(
                'camera_taskboard_frame', f'taskboard_{self.camera_name}'
            ).value
        )

        self.hough_threshold = int(self.declare_parameter('hough_threshold', 40).value)
        self.hough_min_line_length = int(self.declare_parameter('hough_min_line_length', 40).value)
        self.hough_max_line_gap = int(self.declare_parameter('hough_max_line_gap', 12).value)
        self.hough_sample_step_px = int(self.declare_parameter('hough_sample_step_px', 2).value)
        self.min_segment_length_px = float(self.declare_parameter('min_segment_length_px', 40.0).value)
        self.ransac_points_min_total = int(self.declare_parameter('ransac_points_min_total', 200).value)
        self.ransac_max_lines = int(self.declare_parameter('ransac_max_lines', 6).value)
        self.ransac_fit_iterations = int(self.declare_parameter('ransac_fit_iterations', 300).value)
        self.ransac_dist_threshold_px = float(
            self.declare_parameter('ransac_dist_threshold_px', 2.0).value
        )
        self.ransac_min_inliers_base = int(self.declare_parameter('ransac_min_inliers_base', 80).value)
        self.ransac_min_inliers_divisor = int(
            self.declare_parameter('ransac_min_inliers_divisor', 45).value
        )
        self.ransac_outlier_keep_distance_px = float(
            self.declare_parameter('ransac_outlier_keep_distance_px', 3.0).value
        )
        self.ransac_orientation_kmeans_attempts = int(
            self.declare_parameter('ransac_orientation_kmeans_attempts', 5).value
        )
        self.line_fit_norm_epsilon = float(self.declare_parameter('line_fit_norm_epsilon', 1e-6).value)
        self.line_intersection_det_epsilon = float(
            self.declare_parameter('line_intersection_det_epsilon', 1e-6).value
        )
        self.draw_line_length_px = float(self.declare_parameter('draw_line_length_px', 3000.0).value)
        self.pose_axis_length_m = float(self.declare_parameter('pose_axis_length_m', 0.15).value)
        self.publish_debug_visualization = bool(
            self.declare_parameter('publish_debug_visualization', True).value
        )
        self.color_logo_max_age_sec = float(
            self.declare_parameter('color_logo_max_age_sec', 0.3).value
        )
        sensor_qos_depth = max(1, int(self.declare_parameter('sensor_qos_depth', 1).value))
        publisher_qos_depth = max(1, int(self.declare_parameter('publisher_qos_depth', 10).value))
        image_qos_depth = max(1, int(self.declare_parameter('image_qos_depth', 10).value))

        short_edge = float(self.declare_parameter('taskboard_short_edge_m', 0.30).value)
        long_edge = float(self.declare_parameter('taskboard_long_edge_m', 0.42).value)
        # TL, TR, BR, BL: +X is the short edge, +Y is the long edge, +Z is board normal.
        self.taskboard_3d_points = np.array(
            [
                [-short_edge / 2, long_edge / 2, 0],
                [short_edge / 2, long_edge / 2, 0],
                [short_edge / 2, -long_edge / 2, 0],
                [-short_edge / 2, -long_edge / 2, 0],
            ],
            dtype=np.float32,
        )

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.use_camera_info_frame = bool(
            self.declare_parameter('use_camera_info_frame', True).value
        )
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self._logo_msg = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=sensor_qos_depth,
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=image_qos_depth,
        )

        self.create_subscription(Image, self.canny_image_topic, self.image_callback, image_qos)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.create_subscription(
            PointStamped, self.color_logo_center_topic, self.logo_center_callback, image_qos
        )
        self.debug_publisher = None
        if self.publish_debug_visualization:
            self.debug_publisher = self.create_publisher(
                Image, self.debug_image_topic, publisher_qos_depth
            )

        self.get_logger().info(f'RANSAC/Hough taskboard detection started for {self.camera_name}')
        self.get_logger().info(f'Subscribing to Canny image {self.canny_image_topic}')
        self.get_logger().info(f'Subscribing to camera info {self.camera_info_topic}')
        self.get_logger().info(
            f'Subscribing to color logo center {self.color_logo_center_topic} '
            f'(max age {self.color_logo_max_age_sec:.2f}s)'
        )
        self.get_logger().info(
            f'Publishing camera pose {self.camera_optical_frame} -> {self.camera_taskboard_frame}'
        )
        if self.publish_debug_visualization:
            self.get_logger().info(f'Publishing debug image on {self.debug_image_topic}')

    def camera_info_callback(self, msg):
        if self.camera_info_received:
            return
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.camera_info_received = True
        if self.use_camera_info_frame and msg.header.frame_id:
            self.camera_optical_frame = msg.header.frame_id
        self.get_logger().info(
            f'{self.camera_name} calibration received: '
            f'fx={self.camera_matrix[0, 0]:.2f}, fy={self.camera_matrix[1, 1]:.2f}, '
            f'frame={self.camera_optical_frame}'
        )

    def logo_center_callback(self, msg):
        self._logo_msg = msg

    def _recent_logo_center(self, image_stamp):
        if self._logo_msg is None:
            return None
        logo_stamp = self._logo_msg.header.stamp
        if logo_stamp.sec == 0 and logo_stamp.nanosec == 0:
            return None
        age_sec = abs((Time.from_msg(image_stamp) - Time.from_msg(logo_stamp)).nanoseconds) / 1e9
        if age_sec > self.color_logo_max_age_sec:
            return None
        return np.array([self._logo_msg.point.x, self._logo_msg.point.y], dtype=np.float32)

    def image_callback(self, msg):
        try:
            edges = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            corners, detected_lines = self.detect_taskboard_corners_ransac(edges)

            debug_image = None
            if self.publish_debug_visualization:
                debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                for line in detected_lines:
                    self.draw_infinite_line(debug_image, line, (0, 180, 255), 2)

            if corners is None:
                if debug_image is not None:
                    cv2.putText(
                        debug_image,
                        'RANSAC rectangle not found',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 165, 255),
                        2,
                    )
                self._publish_debug(msg, debug_image)
                return

            logo_center = self._recent_logo_center(msg.header.stamp)
            image_points = self.align_corners_to_model_axes(corners, logo_center)
            if debug_image is not None:
                corner_poly = np.round(image_points).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug_image, [corner_poly], True, (0, 255, 0), 3)
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for i, corner in enumerate(image_points):
                    x, y = int(corner[0]), int(corner[1])
                    cv2.circle(debug_image, (x, y), 8, (255, 0, 0), -1)
                    cv2.putText(
                        debug_image,
                        corner_labels[i],
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                if logo_center is not None:
                    lx, ly = int(logo_center[0]), int(logo_center[1])
                    cv2.circle(debug_image, (lx, ly), 10, (255, 0, 255), 2)
                    cv2.putText(
                        debug_image,
                        'logo',
                        (lx + 12, ly - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )

            if not self.camera_info_received:
                self.get_logger().warn(
                    f'{self.camera_name}: corners found but camera_info not received yet',
                    throttle_duration_sec=2.0,
                )
                self._publish_debug(msg, debug_image)
                return

            success, rvec, tvec = cv2.solvePnP(
                self.taskboard_3d_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not success:
                self.get_logger().warn(
                    f'{self.camera_name} PnP failed to converge',
                    throttle_duration_sec=2.0,
                )
                self._publish_debug(msg, debug_image)
                return

            self.publish_camera_taskboard_transform(msg.header, rvec, tvec)

            distance = float(np.linalg.norm(tvec))
            if debug_image is not None:
                self._draw_pose_overlay(debug_image, image_points, rvec, tvec, distance)

            self.get_logger().info(
                f'{self.camera_name} taskboard pose: '
                f'pos=({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}), '
                f'dist={distance:.3f}m',
                throttle_duration_sec=1.0,
            )
            self._publish_debug(msg, debug_image)
        except Exception as e:
            self.get_logger().error(f'Error processing {self.camera_name} Canny image: {e}')

    def _publish_debug(self, msg, debug_image):
        if debug_image is None or self.debug_publisher is None:
            return
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_publisher.publish(debug_msg)

    def _draw_pose_overlay(self, debug_image, image_points, rvec, tvec, distance):
        axis_length = self.pose_axis_length_m
        axis_points = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, -axis_length],
            ]
        )
        img_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        img_points = img_points.reshape(-1, 2).astype(int)
        origin = tuple(img_points[0])
        cv2.line(debug_image, origin, tuple(img_points[1]), (0, 0, 255), 3)
        cv2.line(debug_image, origin, tuple(img_points[2]), (0, 255, 0), 3)
        cv2.line(debug_image, origin, tuple(img_points[3]), (255, 0, 0), 3)

        projected_corners, _ = cv2.projectPoints(
            self.taskboard_3d_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        projected_corners = projected_corners.reshape(-1, 2).astype(int)
        cv2.polylines(debug_image, [projected_corners], True, (255, 255, 0), 2)

        center = np.mean(image_points, axis=0).astype(int)
        cx, cy = int(center[0]), int(center[1])
        cv2.putText(
            debug_image,
            f'Dist: {distance:.2f}m',
            (cx - 120, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            debug_image,
            f'Pos: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})m',
            (cx - 120, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    def detect_taskboard_corners_ransac(self, edges):
        """Detect a rectangle from Hough segments, then refine lines with RANSAC."""
        raw_segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if raw_segments is None:
            return None, []

        sampled_points = []
        for seg in raw_segments.reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in seg]
            seg_len = float(np.hypot(x2 - x1, y2 - y1))
            if seg_len < self.min_segment_length_px:
                continue
            sampled_points.append(
                self.sample_segment_points(x1, y1, x2, y2, step=max(1, self.hough_sample_step_px))
            )

        if not sampled_points:
            return None, []

        remaining_points = np.vstack(sampled_points).astype(np.float32)
        if len(remaining_points) < self.ransac_points_min_total:
            return None, []

        all_lines = []
        min_inliers = max(
            self.ransac_min_inliers_base,
            len(remaining_points) // max(1, self.ransac_min_inliers_divisor),
        )
        for _ in range(self.ransac_max_lines):
            if len(remaining_points) < min_inliers:
                break
            line, inlier_mask = self.fit_ransac_line(
                remaining_points,
                iterations=self.ransac_fit_iterations,
                dist_threshold=self.ransac_dist_threshold_px,
                min_inliers=min_inliers,
            )
            if line is None:
                break
            all_lines.append((line, int(np.count_nonzero(inlier_mask))))
            distances = self.line_distance(line, remaining_points)
            remaining_points = remaining_points[distances > self.ransac_outlier_keep_distance_px]

        if len(all_lines) < 4:
            return None, [line for line, _ in all_lines]

        all_lines.sort(key=lambda item: item[1], reverse=True)
        candidate_lines = [line for line, _ in all_lines[:6]]

        angles = np.array([self.line_orientation(line) for line in candidate_lines], dtype=np.float32)
        features = np.column_stack((np.cos(2.0 * angles), np.sin(2.0 * angles))).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        _, cluster_labels, _ = cv2.kmeans(
            features,
            2,
            None,
            criteria,
            self.ransac_orientation_kmeans_attempts,
            cv2.KMEANS_PP_CENTERS,
        )
        cluster_labels = cluster_labels.flatten()

        grouped_lines = []
        for cluster_id in [0, 1]:
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_idx) < 2:
                return None, candidate_lines
            grouped_lines.append([candidate_lines[i] for i in cluster_idx[:2]])

        corners = []
        for line_a in grouped_lines[0]:
            for line_b in grouped_lines[1]:
                pt = self.line_intersection(line_a, line_b)
                if pt is not None:
                    corners.append(pt)

        selected_lines = [line for group in grouped_lines for line in group]
        if len(corners) != 4:
            return None, selected_lines

        corners = self.order_corners(np.array(corners, dtype=np.float32))
        return corners, selected_lines

    def order_corners(self, pts):
        """Order corners as TL, TR, BR, BL from image coordinates."""
        pts = pts.reshape(-1, 2)
        sorted_pts = pts[np.argsort(pts[:, 1])]
        top_pts = sorted_pts[:2]
        bottom_pts = sorted_pts[2:]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
        if len(bottom_pts) == 2:
            return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
        return pts.astype(np.float32)

    def align_corners_to_model_axes(self, image_points, logo_center):
        """Assign TL from a recent magenta logo, else keep short edge as TL->TR.

        The 3D model is TL, TR, BR, BL with +X the short edge and +Y the long edge.
        """
        if image_points is None or len(image_points) < 4:
            return image_points
        if logo_center is not None:
            distances = np.linalg.norm(image_points - logo_center.reshape(1, 2), axis=1)
            return np.roll(image_points, -int(np.argmin(distances)), axis=0)
        top_edge = np.linalg.norm(image_points[1] - image_points[0])
        right_edge = np.linalg.norm(image_points[2] - image_points[1])
        if top_edge > right_edge:
            image_points = np.roll(image_points, 1, axis=0)
        return image_points

    def line_distance(self, line, points):
        a, b, c = line
        return np.abs(a * points[:, 0] + b * points[:, 1] + c)

    def line_from_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = np.hypot(a, b)
        if norm < self.line_fit_norm_epsilon:
            return None
        return np.array([a / norm, b / norm, c / norm], dtype=np.float32)

    def refine_line_fit(self, points):
        if len(points) < 2:
            return None
        vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(np.asarray(vx).reshape(-1)[0])
        vy = float(np.asarray(vy).reshape(-1)[0])
        x0 = float(np.asarray(x0).reshape(-1)[0])
        y0 = float(np.asarray(y0).reshape(-1)[0])
        a = vy
        b = -vx
        c = -(a * x0 + b * y0)
        norm = np.hypot(a, b)
        if norm < self.line_fit_norm_epsilon:
            return None
        return np.array([a / norm, b / norm, c / norm], dtype=np.float32)

    def fit_ransac_line(self, points, iterations=250, dist_threshold=2.5, min_inliers=80):
        if len(points) < 2:
            return None, None

        best_line = None
        best_inlier_mask = None
        best_count = 0
        rng = np.random.default_rng()

        for _ in range(iterations):
            idx = rng.choice(len(points), size=2, replace=False)
            line = self.line_from_points(points[idx[0]], points[idx[1]])
            if line is None:
                continue
            distances = self.line_distance(line, points)
            inlier_mask = distances < dist_threshold
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count > best_count:
                best_count = inlier_count
                best_line = line
                best_inlier_mask = inlier_mask

        if best_line is None or best_count < min_inliers:
            return None, None

        refined_line = self.refine_line_fit(points[best_inlier_mask])
        if refined_line is None:
            return None, None
        refined_inlier_mask = self.line_distance(refined_line, points) < dist_threshold
        if int(np.count_nonzero(refined_inlier_mask)) < min_inliers:
            return None, None
        return refined_line, refined_inlier_mask

    def line_intersection(self, line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        det = a1 * b2 - a2 * b1
        if abs(det) < self.line_intersection_det_epsilon:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return np.array([x, y], dtype=np.float32)

    def line_orientation(self, line):
        a, b, _ = line
        theta = np.arctan2(-a, b)
        if theta < 0:
            theta += np.pi
        return theta

    def draw_infinite_line(self, image, line, color, thickness=2):
        a, b, c = line
        direction = np.array([b, -a], dtype=np.float32)
        base = -c * np.array([a, b], dtype=np.float32)
        p1 = (base + self.draw_line_length_px * direction).astype(np.int32)
        p2 = (base - self.draw_line_length_px * direction).astype(np.int32)
        cv2.line(image, tuple(p1), tuple(p2), color, thickness)

    def sample_segment_points(self, x1, y1, x2, y2, step=2):
        length = float(np.hypot(x2 - x1, y2 - y1))
        n = max(2, int(length // max(1, step)) + 1)
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        return np.column_stack((xs, ys)).astype(np.float32)

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

    def _fill_transform(self, header_stamp, parent_frame, child_frame, rotation, translation):
        quat = self.rotation_matrix_to_quaternion(rotation)
        t = TransformStamped()
        t.header.stamp = header_stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        return t

    def publish_camera_taskboard_transform(self, header, rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation = np.array([tvec[0][0], tvec[1][0], tvec[2][0]], dtype=np.float64)
        t = self._fill_transform(
            header.stamp,
            self.camera_optical_frame,
            self.camera_taskboard_frame,
            rotation_matrix,
            translation,
        )
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    parser = ArgumentParser(add_help=True)
    _, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = RansacHoughTaskboardDetection()
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
