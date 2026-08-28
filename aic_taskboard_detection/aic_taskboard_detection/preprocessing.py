#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Preprocessing(Node):
    def __init__(self):
        super().__init__('preprocessing')

        self.camera_name = str(self.declare_parameter('camera_name', 'center').value).strip()
        self.image_topic = str(
            self.declare_parameter(
                'image_topic', f'/{self.camera_name}_camera/image'
            ).value
        )
        self.blob_image_topic = str(
            self.declare_parameter(
                'blob_image_topic', f'/{self.camera_name}_camera/image_blob'
            ).value
        )
        self.canny_image_topic = str(
            self.declare_parameter(
                'canny_image_topic', f'/{self.camera_name}_camera/image_canny'
            ).value
        )
        self.blob_center_topic = str(
            self.declare_parameter(
                'blob_center_topic', f'/{self.camera_name}_camera/blob_center'
            ).value
        )
        self.color_logo_center_topic = str(
            self.declare_parameter(
                'color_logo_center_topic', f'/{self.camera_name}_camera/color_logo_center'
            ).value
        )
        self.gripper_mask_dir = str(self.declare_parameter('gripper_mask_dir', '').value)

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
        self.canny_threshold_low = int(self.declare_parameter('canny_threshold_low', 50).value)
        self.canny_threshold_high = int(self.declare_parameter('canny_threshold_high', 150).value)
        self.boundary_gradient_kernel_size = int(
            self.declare_parameter('boundary_gradient_kernel_size', 3).value
        )
        self.boundary_exclusion_dilate_kernel_size = int(
            self.declare_parameter('boundary_exclusion_dilate_kernel_size', 5).value
        )
        self.boundary_exclusion_dilate_iterations = int(
            self.declare_parameter('boundary_exclusion_dilate_iterations', 1).value
        )
        self.blob_min_area_px = float(self.declare_parameter('blob_min_area_px', 1500.0).value)
        self.blob_moment_epsilon = float(
            self.declare_parameter('blob_moment_epsilon', 1e-6).value
        )
        self.magenta_lower_hsv = np.array(
            self.declare_parameter('magenta_lower_hsv', [130, 50, 50]).value, dtype=np.uint8
        )
        self.magenta_upper_hsv = np.array(
            self.declare_parameter('magenta_upper_hsv', [170, 255, 255]).value, dtype=np.uint8
        )
        self.image_encoding = str(self.declare_parameter('image_encoding', 'bgr8').value)

        self.bridge = CvBridge()

        mask_file_path = Path(self.gripper_mask_dir) / f'gripper_mask_{self.camera_name}.npy'
        self.gripper_mask = None
        self.gripper_outline_exclusion = None
        try:
            gripper_mask = np.load(str(mask_file_path))
        except (OSError, ValueError) as e:
            self.get_logger().fatal(
                f'Failed to load gripper mask from {mask_file_path}: {e}. '
                'Continuing without a gripper mask; no gripper pixels will be excluded.'
            )
        else:
            self.gripper_mask = cv2.erode(
                gripper_mask,
                self.gripper_mask_dilation_kernel,
                iterations=self.gripper_mask_dilation_iterations,
            )
            self.gripper_outline_exclusion = self._gripper_outline_exclusion(self.gripper_mask)
            self.get_logger().info(
                f'Loaded {self.camera_name} gripper mask from {mask_file_path} '
                f'with dilation kernel={self.gripper_mask_dilation_kernel.shape[0]} '
                f'iterations={self.gripper_mask_dilation_iterations}'
            )

        self.blob_image_pub = self.create_publisher(
            Image, self.blob_image_topic, qos_profile_sensor_data
        )
        self.canny_image_pub = self.create_publisher(
            Image, self.canny_image_topic, qos_profile_sensor_data
        )
        self.blob_center_pub = self.create_publisher(
            PointStamped, self.blob_center_topic, qos_profile_sensor_data
        )
        self.color_logo_center_pub = self.create_publisher(
            PointStamped, self.color_logo_center_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.get_logger().info(f'Subscribing to {self.image_topic}')
        self.get_logger().info(f'Publishing blob image on {self.blob_image_topic}')
        self.get_logger().info(f'Publishing Canny edges on {self.canny_image_topic}')
        self.get_logger().info(f'Publishing blob center on {self.blob_center_topic}')
        self.get_logger().info(f'Publishing color logo center on {self.color_logo_center_topic}')
        self.get_logger().info(f'Gripper mask directory: {self.gripper_mask_dir}')
        self.get_logger().info(f'Preprocessing node started for camera {self.camera_name}.')

    def _gripper_outline_exclusion(self, gripper_mask):
        """Thin band around the gripper outline, where Canny only sees the mask cut."""
        gripper = np.zeros(gripper_mask.shape, dtype=np.uint8)
        gripper[gripper_mask == 0] = 255
        outline = cv2.morphologyEx(
            gripper,
            cv2.MORPH_GRADIENT,
            np.ones(
                (max(1, self.boundary_gradient_kernel_size), max(1, self.boundary_gradient_kernel_size)),
                np.uint8,
            ),
        )
        return cv2.dilate(
            outline,
            np.ones(
                (
                    max(1, self.boundary_exclusion_dilate_kernel_size),
                    max(1, self.boundary_exclusion_dilate_kernel_size),
                ),
                np.uint8,
            ),
            iterations=max(0, self.boundary_exclusion_dilate_iterations),
        )

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

    def find_magenta_center(self, cv_image):
        """Detect the magenta logo in HSV and return its center of mass, or None."""
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        magenta_mask = cv2.inRange(hsv, self.magenta_lower_hsv, self.magenta_upper_hsv)
        if self.gripper_mask is not None:
            magenta_mask = cv2.bitwise_and(magenta_mask, self.gripper_mask)

        moments = cv2.moments(magenta_mask)
        if moments['m00'] <= self.blob_moment_epsilon:
            return None

        cx = float(moments['m10'] / moments['m00'])
        cy = float(moments['m01'] / moments['m00'])
        return (cx, cy)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
            binary = self.to_binary(cv_image)
            if self.gripper_mask is not None:
                binary = cv2.bitwise_and(binary, self.gripper_mask)

            # The taskboard detectors unsubscribe once the board is locked, so skip the
            # edge and logo work while nothing consumes it.
            if self.canny_image_pub.get_subscription_count() > 0:
                edges = cv2.Canny(binary, self.canny_threshold_low, self.canny_threshold_high)
                if self.gripper_outline_exclusion is not None:
                    edges[self.gripper_outline_exclusion > 0] = 0
                canny_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
                canny_msg.header = msg.header
                self.canny_image_pub.publish(canny_msg)

            if self.color_logo_center_pub.get_subscription_count() > 0:
                logo_center = self.find_magenta_center(cv_image)
                if logo_center is not None:
                    cx, cy = logo_center
                    logo_msg = PointStamped()
                    logo_msg.header = msg.header
                    logo_msg.point.x = cx
                    logo_msg.point.y = cy
                    logo_msg.point.z = 0.0
                    self.color_logo_center_pub.publish(logo_msg)

            blob = self.largest_blob(binary)

            if blob is None:
                blob_image = np.zeros_like(binary)
            else:
                blob_image = blob['mask']
                cx, cy = blob['center_px']
                height, width = cv_image.shape[:2]
                center_msg = PointStamped()
                center_msg.header = msg.header
                # Offset from the geometric camera/image center (pixels).
                center_msg.point.x = cx - 0.5 * float(width)
                center_msg.point.y = cy - 0.5 * float(height)
                center_msg.point.z = 0.0
                self.blob_center_pub.publish(center_msg)

            blob_msg = self.bridge.cv2_to_imgmsg(blob_image, encoding='mono8')
            blob_msg.header = msg.header
            self.blob_image_pub.publish(blob_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing {self.camera_name} image: {e}')


def main(args=None):
    parser = ArgumentParser(add_help=True)
    parsed_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    preprocessing = Preprocessing()
    try:
        rclpy.spin(preprocessing)
    except KeyboardInterrupt:
        preprocessing.get_logger().info('Keyboard interrupt. Exiting...')
    finally:
        preprocessing.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
