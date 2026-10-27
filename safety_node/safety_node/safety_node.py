#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.

        # iTTC threshold (seconds) below which we brake. Tune this during testing.
        self.ttc_threshold = 0.8

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

    def odom_callback(self, odom_msg):
        # update current speed from the x component of linear velocity
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # Nothing to brake for if we're basically stationary
        if abs(self.speed) < 1e-3:
            return

        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        n = len(ranges)
        angles = scan_msg.angle_min + np.arange(n) * scan_msg.angle_increment

        # Range rate from projecting velocity onto each beam:
        # r_dot = -v * cos(theta)
        range_rate = -self.speed * np.cos(angles)

        # Only trust valid, in-range readings
        valid = (
            np.isfinite(ranges)
            & (ranges >= scan_msg.range_min)
            & (ranges <= scan_msg.range_max)
        )

        # {-r_dot}+ = max(-r_dot, 0)
        denom = np.maximum(-range_rate, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ittc = np.where(denom > 1e-6, ranges / denom, np.inf)

        ittc = np.where(valid, ittc, np.inf)

        min_ittc = np.min(ittc)

        if min_ittc < self.ttc_threshold:
            self.get_logger().warn(
                f'Collision imminent! min iTTC = {min_ittc:.2f}s. Braking.')
            self.brake()

    def brake(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
