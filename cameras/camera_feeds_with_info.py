import argparse
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import threading
from rclpy.qos import QoSProfile

# Get the directory of the current script (camera_feeds_with_info.py)
script_dir = os.path.dirname(os.path.realpath(__file__))

# Load the camera calibration data from files
def load_calibration_data(calibration_file, camera_side):
    camera_matrix = None
    distortion_coeffs = None
    rectification_matrix = None
    projection_matrix = None
    width = None
    height = None
    
    if not os.path.exists(calibration_file):
        return None, None, None, None, None, None  # Return None values if calibration file doesn't exist

    with open(calibration_file, 'r') as f:
        lines = f.readlines()

    # Extract the calibration data from the narrow_stereo/camera_side section
    section = f'narrow_stereo/{camera_side}'
    inside_section = False

    for i, line in enumerate(lines):
        if line.strip() ==f'width':
            width = int(lines[i + 1].strip())
        elif line.strip() ==f'height':
            height = int(lines[i + 1].strip())
        elif line.strip() == f'[{section}]':
            inside_section = True
        elif inside_section:
            if line.startswith('camera matrix'):
                camera_matrix = []
                for i in range(3):
                    camera_matrix.extend(map(float, lines[lines.index(line) + i + 1].split()))
                camera_matrix = np.array(camera_matrix).reshape(3, 3)

            elif line.startswith('distortion'):
                distortion_coeffs = np.array(list(map(float, lines[lines.index(line) + 1].split())))

            elif line.startswith('rectification'):
                rectification_matrix = []
                for i in range(3):
                    rectification_matrix.extend(map(float, lines[lines.index(line) + i + 1].split()))
                rectification_matrix = np.array(rectification_matrix).reshape(3, 3)

            elif line.startswith('projection'):
                projection_matrix = []
                for i in range(3):
                    projection_matrix.extend(map(float, lines[lines.index(line) + i + 1].split()))
                projection_matrix = np.array(projection_matrix).reshape(3, 4)

            if camera_matrix is not None and distortion_coeffs is not None and rectification_matrix is not None and projection_matrix is not None:
                break

    return camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix, width, height


class VideoCaptureThread(threading.Thread):
    def __init__(self, device, image_topic_name, camera_info_topic_name, node, camera_data, timestamp):
        super().__init__()
        self.device = device
        self.node = node
        self.bridge = CvBridge()
        self.image_publisher = node.create_publisher(Image, image_topic_name, QoSProfile(depth=10))
        self.camera_info_publisher = node.create_publisher(CameraInfo, camera_info_topic_name, QoSProfile(depth=10))

        # Unpack camera_data tuple (camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix)
        self.mtx, self.dist, self.rect, self.proj, self.width, self.height = camera_data

        # Initialize the capture
        self.capture = cv2.VideoCapture(self.device)

        # Set video capture properties based on calibration data (width, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.capture.isOpened():
            self.node.get_logger().error(f"Failed to open video device: {self.device}")
            sys.exit(1)

        self.stop_flag = threading.Event()
        self.timestamp = timestamp  # Store the timestamp passed from the main thread

    def run(self):
        while not self.stop_flag.is_set():
            ret, frame = self.capture.read()
            if ret:
                try:
                    # Publish raw image (no calibration applied to the frame)
                    image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    image_msg.header.stamp = self.timestamp  # Use the shared timestamp
                    image_msg.header.frame_id = "camera_frame"
                    self.image_publisher.publish(image_msg)

                    # If calibration data exists, publish CameraInfo
                    if self.mtx is not None and self.dist is not None and self.rect is not None and self.proj is not None:
                        camera_info_msg = CameraInfo()
                        camera_info_msg.header.stamp = self.timestamp  # Use the shared timestamp
                        camera_info_msg.header.frame_id = "camera_frame"
                        camera_info_msg.width = self.width
                        camera_info_msg.height = self.height
                        camera_info_msg.k = self.mtx.flatten().tolist()  # Camera matrix
                        camera_info_msg.d = self.dist.flatten().tolist()  # Distortion coefficients
                        camera_info_msg.r = self.rect.flatten().tolist()  # Rectification matrix
                        camera_info_msg.p = self.proj.flatten().tolist()  # Projection matrix
                        camera_info_msg.distortion_model = 'plumb_bob'

                        # Publish CameraInfo message if calibration data is available
                        self.camera_info_publisher.publish(camera_info_msg)

                except Exception as e:
                    self.node.get_logger().error(f"Failed to publish frame: {str(e)}")
            else:
                self.node.get_logger().warn(f"Failed to read frame from device {self.device}")

    def stop(self):
        self.stop_flag.set()
        self.capture.release()


class VideoFeedsNode(Node):
    def __init__(self, left_calib_file, right_calib_file, left_device, right_device):
        super().__init__('video_feeds_node')
        self.left_calib_file = left_calib_file
        self.right_calib_file = right_calib_file
        self.get_logger().info(f"Left device: {left_device}, Right device: {right_device}")

        # Load calibration data for both cameras (including width and height)
        self.left_camera_data = load_calibration_data(left_calib_file, 'left')
        self.right_camera_data = load_calibration_data(right_calib_file, 'right')

        # Get the timestamp from the node clock to synchronize the feeds
        self.timestamp = self.get_clock().now().to_msg()

        # Start video capture threads for both cameras
        self.left_capture_thread = VideoCaptureThread(
            left_device, '/left/image_raw', '/left/camera_info', self, self.left_camera_data, self.timestamp)
        self.right_capture_thread = VideoCaptureThread(
            right_device, '/right/image_raw', '/right/camera_info', self, self.right_camera_data, self.timestamp)

        print("starting")
        self.left_capture_thread.start()
        self.right_capture_thread.start()

    def destroy_node(self):
        # Stop all capture threads
        self.left_capture_thread.stop()
        self.right_capture_thread.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Create argument parser
    parser = argparse.ArgumentParser(description="ROS 2 Video Feeds Node")
    parser.add_argument('--left_device', type=str, default='', help='Left camera device path')
    parser.add_argument('--right_device', type=str, default='', help='Right camera device path')

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Check if either left_device or right_device are empty
    if not parsed_args.left_device or not parsed_args.right_device:
        print("Error: Both --left_device and --right_device must be specified.")
        sys.exit(1)  # Stop execution

    # Get the script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Set the paths for left.ini and right.ini relative to the current script location
    left_calib_file = os.path.join(script_dir, 'left.ini')
    right_calib_file = os.path.join(script_dir, 'right.ini')

    # Create the node with the passed devices
    node = VideoFeedsNode(left_calib_file, right_calib_file, parsed_args.left_device, parsed_args.right_device)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

