import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import threading
from rclpy.qos import QoSProfile

class VideoCaptureThread(threading.Thread):
    def __init__(self, device, topic_name, node):
        super().__init__()
        self.device = device
        self.node = node
        self.bridge = CvBridge()
        self.publisher = node.create_publisher(Image, topic_name, QoSProfile(depth=100))
        self.capture = cv2.VideoCapture(self.device)

        # Set camera properties (ensure this matches desired frame rate/resolution)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.capture.isOpened():
            self.node.get_logger().error(f"Failed to open video device: {self.device}")
            sys.exit(1)

        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            ret, frame = self.capture.read()
            if ret:
                try:
                    image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    self.publisher.publish(image_msg)
                except Exception as e:
                    self.node.get_logger().error(f"Failed to publish frame: {str(e)}")
            else:
                self.node.get_logger().warn(f"Failed to read frame from device {self.device}")

    def stop(self):
        self.stop_flag.set()
        self.capture.release()

class VideoFeedsNode(Node):
    def __init__(self, video_devices):
        super().__init__('video_feeds_node')
        self.video_devices = video_devices
        self.capture_threads = []

        for index, device in enumerate(self.video_devices):
            topic_name = f'/camera/device_{index}/image_raw'  # Each camera has its own topic
            capture_thread = VideoCaptureThread(device, topic_name, self)
            capture_thread.start()
            self.capture_threads.append(capture_thread)

    def destroy_node(self):
        # Stop all capture threads
        for thread in self.capture_threads:
            thread.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # Initialize video devices (replace with actual device paths)
    video_devices = ["/dev/video2", "/dev/video4"]

    node = VideoFeedsNode(video_devices)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
