#!/usr/bin/env python3

"""Detect SC ports or NIC ports in a wrist-camera image.

One node, two components:
  component=sc_port  method=hsv   HSV blue-blob detection
  component=sc_port  method=yolo  YOLO-pose (bbox + image-plane yaw)
  component=nic_port method=yolo  YOLO bbox detection of NIC ports

Model loading and visualization follow the same layout for both YOLO
backends: ``model_path`` / ``confidence_threshold`` / ``device`` / ``imgsz``.
HSV is the extra backend for the SC port when a trained model is not used.

Publishes:
  ~/detected (std_msgs/Bool)
  ~/detections (vision_msgs/Detection2DArray)
  /{camera}_camera/image_{component}s (sensor_msgs/Image) when enabled
  ~/port_pixel_centers (geometry_msgs/PoseArray) for nic_port
"""

import math
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

VALID_COMPONENTS = ('sc_port', 'nic_port')
VALID_METHODS = ('hsv', 'yolo')

# BGR colors per class. Match the original SC-port viz so training and
# deployed overlays read the same at a glance.
CLASS_COLORS = {
    0: (0, 200, 0),
    1: (0, 128, 255),
}
NEUTRAL = (200, 200, 200)
YAW_ARROW_SCALE = 4.0


class ComponentDetection(Node):
    """Detect an SC port (HSV or YOLO) or NIC ports (YOLO) from a camera image."""

    def __init__(self):
        super().__init__('component_detection')

        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        self.image_topic = str(
            self.declare_parameter(
                'image_topic', f'/{self.camera_name}_camera/image'
            ).value
        )
        self.component = str(self.declare_parameter('component', 'sc_port').value).strip()
        self.method = str(self.declare_parameter('method', 'yolo').value).strip()
        self.visualization_topic = str(
            self.declare_parameter(
                'visualization_topic',
                f'/{self.camera_name}_camera/image_{self.component}s',
            ).value
        )
        self.publish_viz = bool(self.declare_parameter('publish_visualization', False).value)
        self.image_encoding = str(self.declare_parameter('image_encoding', 'bgr8').value)
        self.hsv_lower = np.array(
            self.declare_parameter('hsv_lower', [95, 40, 80]).value, dtype=np.uint8
        )
        self.hsv_upper = np.array(
            self.declare_parameter('hsv_upper', [135, 255, 255]).value, dtype=np.uint8
        )
        self.min_contour_area = float(self.declare_parameter('min_contour_area', 50.0).value)
        self.morph_kernel_size = max(1, int(self.declare_parameter('morph_kernel_size', 5).value))
        self.class_id = str(self.declare_parameter('class_id', 'sc_port').value)
        self.conf_thresh = float(self.declare_parameter('confidence_threshold', 0.5).value)
        device = str(self.declare_parameter('device', '').value)
        self.device = device if device else None
        self.imgsz = int(self.declare_parameter('imgsz', 640).value)
        model_path = str(self.declare_parameter('model_path', '').value)
        self.port_class_ids = {
            int(v) for v in self.declare_parameter('port_class_ids', [0, 1]).value
        }

        self._validate_backend()

        self.model = None
        if self.method == 'yolo':
            self.model = self._load_yolo(self._resolve_model_path(model_path))

        self.bridge = CvBridge()

        self.detected_pub = self.create_publisher(Bool, '~/detected', qos_profile_sensor_data)
        self.detections_pub = self.create_publisher(
            Detection2DArray, '~/detections', qos_profile_sensor_data
        )
        self.port_pixels_pub = None
        if self.component == 'nic_port':
            self.port_pixels_pub = self.create_publisher(
                PoseArray, '~/port_pixel_centers', qos_profile_sensor_data
            )
        self.viz_pub = None
        if self.publish_viz:
            self.viz_pub = self.create_publisher(
                Image, self.visualization_topic, qos_profile_sensor_data
            )

        self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Component detection ready: component={self.component} method={self.method}'
        )
        self.get_logger().info(f'Subscribing to {self.image_topic}')
        if self.publish_viz:
            self.get_logger().info(f'Publishing visualization on {self.visualization_topic}')
        if self.method == 'yolo':
            self.get_logger().info(
                f'YOLO conf>={self.conf_thresh} imgsz={self.imgsz} '
                f'device={device or "auto"} viz={"on" if self.publish_viz else "off"}'
            )
        else:
            self.get_logger().info(
                f'HSV lower={self.hsv_lower.tolist()} upper={self.hsv_upper.tolist()} '
                f'min_area={self.min_contour_area} viz={"on" if self.publish_viz else "off"}'
            )

    def _validate_backend(self):
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Unknown component '{self.component}'. "
                f"Expected one of: {', '.join(VALID_COMPONENTS)}"
            )
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Expected one of: {', '.join(VALID_METHODS)}"
            )
        if self.component == 'nic_port' and self.method != 'yolo':
            raise ValueError('NIC port detection only supports method=yolo')

    def _resolve_model_path(self, model_path):
        if model_path:
            return Path(model_path)
        share = Path(get_package_share_directory('aic_component_detection'))
        return share / 'model' / f'{self.component}.pt'

    def _load_yolo(self, model_path):
        if not model_path.is_file():
            self.get_logger().fatal(
                f"Model weights not found: '{model_path}'. Set the 'model_path' parameter."
            )
            raise FileNotFoundError(f'Model weights not found: {model_path}')
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal('ultralytics not installed. Run: pip install ultralytics')
            raise

        model = YOLO(str(model_path))
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        model(
            dummy,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        self.get_logger().info(
            f'YOLO model loaded from: {model_path} '
            f'(classes={getattr(model, "names", None)})'
        )
        return model

    def _class_name(self, class_id):
        names = getattr(self.model, 'names', None) if self.model is not None else None
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        if self.method == 'hsv':
            return self.class_id
        return f'{self.component}_{class_id}'

    def image_callback(self, msg):
        try:
            scene_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        try:
            if self.method == 'hsv':
                overlays = self._detect_hsv(scene_color)
            else:
                overlays = self._detect_yolo(scene_color)
        except Exception as exc:
            self.get_logger().error(f'Detection failed: {exc}')
            return

        detected_msg = Bool()
        detected_msg.data = len(overlays) > 0
        self.detected_pub.publish(detected_msg)

        det_array = Detection2DArray()
        det_array.header = msg.header
        for overlay in overlays:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (overlay['x1'] + overlay['x2']) / 2.0
            det.bbox.center.position.y = (overlay['y1'] + overlay['y2']) / 2.0
            det.bbox.center.theta = overlay['yaw_rad']
            det.bbox.size_x = overlay['x2'] - overlay['x1']
            det.bbox.size_y = overlay['y2'] - overlay['y1']
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = overlay['class_name']
            hyp.hypothesis.score = overlay['conf']
            det.results.append(hyp)
            det_array.detections.append(det)
        self.detections_pub.publish(det_array)

        if self.port_pixels_pub is not None:
            port_pixels = PoseArray()
            port_pixels.header = msg.header
            entries = []
            for overlay in overlays:
                if overlay['cls_id'] not in self.port_class_ids:
                    continue
                u = (overlay['x1'] + overlay['x2']) / 2.0
                v = (overlay['y1'] + overlay['y2']) / 2.0
                entries.append((u, v))
            entries.sort(key=lambda t: t[0])
            for u, v in entries:
                pose = Pose()
                pose.position.x = u
                pose.position.y = v
                pose.position.z = 0.0
                pose.orientation.w = 1.0
                port_pixels.poses.append(pose)
            self.port_pixels_pub.publish(port_pixels)

        if overlays:
            self.get_logger().debug(
                'detected '
                + ', '.join(
                    f"{o['class_name']}@{o['conf']:.2f} "
                    f"theta={math.degrees(o['yaw_rad']):+.1f}deg"
                    for o in overlays
                )
            )
        else:
            self.get_logger().debug(f'No {self.component} detected in this frame')

        if self.viz_pub is not None:
            annotated = self._draw_visualization(scene_color, overlays)
            viz_msg = self.bridge.cv2_to_imgmsg(annotated, encoding=self.image_encoding)
            viz_msg.header = msg.header
            self.viz_pub.publish(viz_msg)

    def _detect_hsv(self, scene_color):
        hsv = cv2.cvtColor(scene_color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        img_area = float(scene_color.shape[0] * scene_color.shape[1])
        overlays = []
        for contour in valid:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            score = min(1.0, area / img_area) if img_area > 0 else 0.0
            overlays.append(
                {
                    'x1': float(x),
                    'y1': float(y),
                    'x2': float(x + w),
                    'y2': float(y + h),
                    'cls_id': 0,
                    'class_name': self.class_id,
                    'conf': score,
                    'yaw_rad': 0.0,
                    'anchor_uv': None,
                    'axis_uv': None,
                }
            )
        return overlays

    def _detect_yolo(self, scene_color):
        results = self.model(
            scene_color,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        all_kpts_xy = None
        keypoints = result.keypoints
        if keypoints is not None and keypoints.xy is not None:
            try:
                all_kpts_xy = keypoints.xy.cpu().numpy()
            except Exception:
                all_kpts_xy = None

        overlays = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            anchor_uv = None
            axis_uv = None
            yaw_rad = 0.0
            if (
                all_kpts_xy is not None
                and i < len(all_kpts_xy)
                and len(all_kpts_xy[i]) >= 2
            ):
                anchor_uv = (float(all_kpts_xy[i][0][0]), float(all_kpts_xy[i][0][1]))
                axis_uv = (float(all_kpts_xy[i][1][0]), float(all_kpts_xy[i][1][1]))
                du = axis_uv[0] - anchor_uv[0]
                dv = axis_uv[1] - anchor_uv[1]
                if abs(du) > 1e-9 or abs(dv) > 1e-9:
                    yaw_rad = float(math.atan2(dv, du))
            overlays.append(
                {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'cls_id': cls_id,
                    'class_name': self._class_name(cls_id),
                    'conf': conf,
                    'yaw_rad': yaw_rad,
                    'anchor_uv': anchor_uv,
                    'axis_uv': axis_uv,
                }
            )
        return overlays

    def _draw_visualization(self, bgr, overlays):
        img = bgr.copy()
        height, width = img.shape[:2]
        for overlay in overlays:
            color = CLASS_COLORS.get(overlay['cls_id'], NEUTRAL)
            p1 = (int(round(overlay['x1'])), int(round(overlay['y1'])))
            p2 = (int(round(overlay['x2'])), int(round(overlay['y2'])))
            cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)

            if overlay['anchor_uv'] is not None:
                label = (
                    f"{overlay['class_name']} {overlay['conf']:.2f}  "
                    f"{math.degrees(overlay['yaw_rad']):+.1f}deg"
                )
            else:
                label = f"{overlay['class_name']} {overlay['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tx = max(5, min(width - tw - 8, p1[0]))
            ty = max(th + 6, p1[1] - 6)
            cv2.rectangle(img, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), color, -1)
            cv2.putText(
                img,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            anchor = overlay['anchor_uv']
            axis = overlay['axis_uv']
            if anchor is not None:
                au = int(round(anchor[0]))
                av = int(round(anchor[1]))
                cv2.drawMarker(img, (au, av), color, cv2.MARKER_CROSS, 16, 2)
                cv2.circle(img, (au, av), 5, color, 1, cv2.LINE_AA)
                if axis is not None:
                    xu = int(round(axis[0]))
                    xv = int(round(axis[1]))
                    cv2.drawMarker(img, (xu, xv), color, cv2.MARKER_DIAMOND, 12, 2)
                    du = xu - au
                    dv = xv - av
                    tip = (
                        int(round(au + du * YAW_ARROW_SCALE)),
                        int(round(av + dv * YAW_ARROW_SCALE)),
                    )
                    cv2.arrowedLine(
                        img, (au, av), tip, color, 2, cv2.LINE_AA, tipLength=0.2
                    )
            else:
                cx = int(round((overlay['x1'] + overlay['x2']) / 2.0))
                cy = int(round((overlay['y1'] + overlay['y2']) / 2.0))
                cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        return img


def main(args=None):
    parser = ArgumentParser(add_help=True)
    _, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    try:
        node = ComponentDetection()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f'[FATAL] {exc}')
        if rclpy.ok():
            rclpy.shutdown()
        return

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
