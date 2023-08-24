#!/usr/bin/env python
import roslib
roslib.load_manifest('mico_octomap') 
import rospy
import tf
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
# import sys
# # adding python directory to the system path
# sys.path.insert(0, '/home/stephanie/mico_ws/src/python')

from pose_estimation import intrinsics, get_chess_corners

# based on http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPytho

CHESS = [0.095, -0.125, -0.037] # hardcoded translation from bottom right chess to mico

class calibration_node:
    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)
        
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.depth_callback)
        self.depth_sub = rospy.Subscriber("/camera/color/camera_info",CameraInfo,self.camera_callback)

        self.br = tf.TransformBroadcaster()

        self.tvec = None
        self.r = None

        self.color = None
        self.depth = None
        self.camera_intrinsics = None


    def color_callback(self,data):
        try:
            self.color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def depth_callback(self,data):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)

    def camera_callback(self, data):
        if self.camera_intrinsics: 
            return
        self.camera_intrinsics = intrinsics()
        self.camera_intrinsics.ppx = data.K[2]
        self.camera_intrinsics.ppy = data.K[5]
        self.camera_intrinsics.fx = data.K[0]
        self.camera_intrinsics.fy = data.K[4]
        
    
    def perform_calibration(self):
        # do stuff here
        while not rospy.is_shutdown():
            # once we load frames
            if self.color is not None and self.depth is not None and self.camera_intrinsics is not None:
                try:
                    rmat, tvec = get_chess_corners(self.color, self.depth, self.camera_intrinsics)
                    # wrap rmat in 4x4 matrix
                    r = np.identity(4)
                    r[:3, :3] = rmat
                    self.tvec = tvec
                    self.r = r
                    self.chess = np.array(CHESS)
                except Exception as e:
                    rospy.logwarn(e)
                    
                if self.tvec is not None: # once we have a saved transform
                    # rospy.loginfo(self.tvec)
                    # rospy.loginfo(tf.transformations.quaternion_from_matrix(self.r))
                    self.br.sendTransform(self.tvec,
                                tf.transformations.quaternion_from_matrix(self.r),
                                rospy.Time.now(),
                                "camera_link", #from child
                                "chess") #to parent
                    self.br.sendTransform(self.chess,
                                          tf.transformations.quaternion_from_matrix(np.identity(4)),
                                          rospy.Time.now(),
                                          "chess",
                                          "world")
            rospy.Rate(1).sleep()

if __name__ == '__main__':
    n = calibration_node()
    rospy.init_node('calibration')
    n.perform_calibration()
    # cv2.destroyAllWindows()
    
