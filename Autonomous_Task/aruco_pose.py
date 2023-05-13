#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import argparse  # added import statement for argparse
from sensor_msgs.msg import Image
bridge = CvBridge()

def image_callback(image_msg, args):
    bridge, pose_pub, namespace = args
    detect_aruco(image_msg, bridge, pose_pub, namespace)

def detect_aruco(image_msg, bridge, pose_pub, args):
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    camera_matrix = np.array([[628.158, 0., 324.099], [0., 628.156, 260.908], [0., 0., 1.]])
    dist_coeffs = np.array([0.0995485, -0.206384, 0.00754589, 0.00336531, 0])
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_image, dictionary, parameters=parameters)

    if marker_ids is not None and args.marker_id in marker_ids:
        idx = np.where(marker_ids == args.marker_id)[0][0]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, args.marker_length, camera_matrix, dist_coeffs)
        marker_center = tvecs[idx][0] + np.array([0, 0, args.marker_length / 2.0])
        header = Header(stamp=image_msg.header.stamp, frame_id='camera_frame')
        pose = PoseStamped(header=header)
        pose.pose.position.x = marker_center[0]
        pose.pose.position.y = marker_center[1]
        pose.pose.position.z = marker_center[2]
        q = cv2.Rodrigues(rvecs[idx])[0]
        pose.pose.orientation.x = q[0, 0]
        pose.pose.orientation.y = q[1, 0]
        pose.pose.orientation.z = q[2, 0]
        pose.pose.orientation.w = np.sqrt(1 + q[0, 0] + q[1, 1] + q[2, 2]) / 2.0
        pose_pub.publish(pose)
        # Add x, y, z text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 0, 255)
        x, y, z = marker_center
        x_text = f'X: {x:.2f}'
        y_text = f'Y: {y:.2f}'
        z_text = f'Z: {z:.2f}'
        cv2.putText(cv_image, x_text, (10, 20), font, font_scale, (0, 0, 255), thickness)
        cv2.putText(cv_image, y_text, (10, 40), font, font_scale, (255, 0, 0), thickness)
        cv2.putText(cv_image, z_text, (10, 60), font, font_scale, (0, 255, 0), thickness)
        # Display the annotated video stream
        cv2.imshow("Annotated Video Stream", cv_image)
        cv2.waitKey(1)

    # Convert OpenCV image to ROS image message
    image_msg_out = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    return image_msg_out

if __name__ == '__main__':
    # Define ROS node name
    rospy.init_node('pose_estimator')
    parser = argparse.ArgumentParser(description='Detect ArUco markers and estimate their poses')
    parser.add_argument('--marker_id', type=int, default=0, help='ID of marker to detect')
    parser.add_argument('--marker_length', type=float, default=0.1, help='Length of marker in meters')
    args = parser.parse_args()
    pose_topic = '/marker_pose'
    pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=10)
    image_topic = '/image_topic_2'
    image_sub = rospy.Subscriber(image_topic, Image, image_callback, (bridge, pose_pub, args))
    rospy.spin()
