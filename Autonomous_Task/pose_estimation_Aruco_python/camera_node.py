#!/usr/bin/env python3
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
  def __init__(self):
    print ('here')
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)
    self.bridge = CvBridge()
    self.callback()

  def callback(self):
    while img1.isOpened():
      succ, frame = img1.read()
      cv2.imshow('image', frame)
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
      cv2.waitKey(1)


if __name__ == '__main__':
    img1 = cv2.VideoCapture(0)
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

