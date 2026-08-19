#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class BlobDetection(Node):
    def __init__(self):
        super().__init__('blob_detection')

        self.camera_names = [
            str(c).strip()
            for c in self.declare_parameter('camera_names', ['center', 'left', 'right']).value
        ]
        self.camera_image_topic_suffix = str(
            self.declare_parameter('camera_image_topic_suffix', '_camera/image').value
        )
        self.camera_blob_output_topic_suffix = str(
            self.declare_parameter('camera_blob_output_topic_suffix', '_camera/image_blob').value
        )
        self.blob_center_topic = str(
            self.declare_parameter('blob_center_topic', '/center_camera/blob_center').value
        )
        self.blob_center_camera = str(
            self.declare_parameter('blob_center_camera', 'center').value
        )
        self.gripper_mask_dir = str(
            self.declare_parameter('gripper_mask_dir', '').value
        )

        self.gray_blur_kernel_size = int(self.declare_parameter('gray_blur_kernel_size', 5).value)
        self.binary_morph_kernel_size = int(
            self.declare_parameter('binary_morph_kernel_size', 5).value
        )
        self.binary_erode_iterations = int(
            self.declare_parameter('binary_erode_iterations', 2).value
        )
        self.binary_dilate_iterations = int(
            self.declare_parameter('binary_dilate_iterations', 1).value
        )
        self.gripper_mask_dilation_kernel_size = int(
            self.declare_parameter('gripper_mask_dilation_kernel_size', 9).value
        )
        self.gripper_mask_dilation_iterations = int(
            self.declare_parameter('gripper_mask_dilation_iterations', 2).value
        )
        self.gripper_mask_dilation_kernel = np.ones(
            (
                max(1, self.gripper_mask_dilation_kernel_size),
                max(1, self.gripper_mask_dilation_kernel_size),
            ),
            dtype=np.uint8,
        )
        self.blob_min_area_px = float(self.declare_parameter('blob_min_area_px', 1500.0).value)
        self.blob_moment_epsilon = float(
            self.declare_parameter('blob_moment_epsilon', 1e-6).value
        )
        self.image_encoding = str(self.declare_parameter('image_encoding', 'bgr8').value)
        sensor_qos_depth = max(1, int(self.declare_parameter('sensor_qos_depth', 1).value))
        publisher_qos_depth = max(1, int(self.declare_parameter('publisher_qos_depth', 10).value))

        self.bridge = CvBridge()
        self.image_subscriptions = []
        self.camera_states = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=sensor_qos_depth,
        )

        self.blob_center_pub = self.create_publisher(
            PointStamped, self.blob_center_topic, publisher_qos_depth
        )

        for camera_name in self.camera_names:
            image_topic = f'/{camera_name}{self.camera_image_topic_suffix}'
            image_sub = self.create_subscription(
                Image,
                image_topic,
                lambda msg, cam=camera_name: self.image_callback(msg, cam),
                sensor_qos,
            )
            self.image_subscriptions.append(image_sub)

            mask_file_path = Path(self.gripper_mask_dir) / f'gripper_mask_{camera_name}.npy'
            gripper_mask = np.load(str(mask_file_path))
            gripper_mask = cv2.erode(
                gripper_mask,
                self.gripper_mask_dilation_kernel,
                iterations=self.gripper_mask_dilation_iterations,
            )
            self.get_logger().info(
                f'Loaded {camera_name} gripper mask from {mask_file_path} '
                f'with dilation kernel={self.gripper_mask_dilation_kernel.shape[0]} '
                f'iterations={self.gripper_mask_dilation_iterations}'
            )

            self.camera_states[camera_name] = {
                'gripper_mask': gripper_mask,
                'blob_publisher': self.create_publisher(
                    Image,
                    f'/{camera_name}{self.camera_blob_output_topic_suffix}',
                    publisher_qos_depth,
                ),
            }
            self.get_logger().info(f'Subscribing to {image_topic}')
            self.get_logger().info(
                f'Publishing blob image on /{camera_name}{self.camera_blob_output_topic_suffix}'
            )

        self.get_logger().info(
            f'Publishing {self.blob_center_camera} blob center on {self.blob_center_topic}'
        )
        self.get_logger().info(f'Gripper mask directory: {self.gripper_mask_dir}')
        self.get_logger().info('Blob detection node started.')

    def to_binary(self, cv_image):
        """Build an inverted Otsu binary image of dark objects."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur_kernel = max(1, self.gray_blur_kernel_size)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        morph_kernel = np.ones(
            (max(1, self.binary_morph_kernel_size), max(1, self.binary_morph_kernel_size)),
            np.uint8,
        )
        binary = cv2.erode(binary, morph_kernel, iterations=max(0, self.binary_erode_iterations))
        binary = cv2.dilate(binary, morph_kernel, iterations=max(0, self.binary_dilate_iterations))
        return binary

    def largest_blob(self, binary):
        """Return the largest contour's mask and centroid, or None if too small."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < self.blob_min_area_px:
            return None

        moments = cv2.moments(largest)
        if moments['m00'] <= self.blob_moment_epsilon:
            return None

        cx = float(moments['m10'] / moments['m00'])
        cy = float(moments['m01'] / moments['m00'])
        blob_mask = np.zeros_like(binary)
        cv2.drawContours(blob_mask, [largest], contourIdx=-1, color=255, thickness=cv2.FILLED)
        return {
            'center_px': (cx, cy),
            'area': area,
            'mask': blob_mask,
        }

    def image_callback(self, msg, camera_name):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
            binary = self.to_binary(cv_image)
            gripper_mask = self.camera_states[camera_name]['gripper_mask']
            if gripper_mask is not None:
                binary = cv2.bitwise_and(binary, gripper_mask)
            blob = self.largest_blob(binary)

            if blob is None:
                blob_image = np.zeros_like(binary)
            else:
                blob_image = blob['mask']
                if camera_name == self.blob_center_camera:
                    cx, cy = blob['center_px']
                    center_msg = PointStamped()
                    center_msg.header = msg.header
                    center_msg.point.x = cx
                    center_msg.point.y = cy
                    center_msg.point.z = 0.0
                    self.blob_center_pub.publish(center_msg)

            blob_msg = self.bridge.cv2_to_imgmsg(blob_image, encoding='mono8')
            blob_msg.header = msg.header
            self.camera_states[camera_name]['blob_publisher'].publish(blob_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing {camera_name} image: {e}')


def main(args=None):
    parser = ArgumentParser(add_help=True)
    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    blob_detection = BlobDetection()
    try:
        rclpy.spin(blob_detection)
    except KeyboardInterrupt:
        blob_detection.get_logger().info('Keyboard interrupt. Exiting...')
    finally:
        blob_detection.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
