#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations
import numpy as np
import os
import yaml

class TransformPublisher:
    def __init__(self):
        rospy.init_node('transform_publisher')

        # Create a static transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        #! TF: ARUCO_BASE_LINK
        aruco_robot_tf = TransformStamped()
        aruco_robot_tf.header.frame_id = "base_link"
        aruco_robot_tf.child_frame_id = "aruco"
        quaternion = tf.transformations.quaternion_from_euler(0, 25.8*np.pi/180, -180*np.pi/180)
        aruco_robot_tf.transform.translation.x = -0.025
        aruco_robot_tf.transform.translation.y = 0.31
        aruco_robot_tf.transform.translation.z = 0.0217
        aruco_robot_tf.transform.rotation.x = quaternion[0]
        aruco_robot_tf.transform.rotation.y = quaternion[1]
        aruco_robot_tf.transform.rotation.z = quaternion[2]
        aruco_robot_tf.transform.rotation.w = quaternion[3]

        #! TF: ARUCO_OPTICAL_FRAME
        aruco_camera_mtx = np.loadtxt("/home/drx/catkin_ws/src/ur5_interface/data/aruco_camera.txt")
        aruco_camera_mtx = np.linalg.inv(aruco_camera_mtx)
        aruco_camera_tf = TransformStamped()
        aruco_camera_tf.header.frame_id = "aruco"
        aruco_camera_tf.child_frame_id = "camera_link"
        quaternion = tf.transformations.quaternion_from_matrix(aruco_camera_mtx)
        aruco_camera_tf.transform.translation.x = aruco_camera_mtx[0,3]
        aruco_camera_tf.transform.translation.y = aruco_camera_mtx[1,3]
        aruco_camera_tf.transform.translation.z = aruco_camera_mtx[2,3]
        aruco_camera_tf.transform.rotation.x = quaternion[0]
        aruco_camera_tf.transform.rotation.y = quaternion[1]
        aruco_camera_tf.transform.rotation.z = quaternion[2]
        aruco_camera_tf.transform.rotation.w = quaternion[3]

        while not rospy.is_shutdown():
            aruco_camera_tf.header.stamp = rospy.Time.now()
            aruco_robot_tf.header.stamp = rospy.Time.now()

            self.tf_broadcaster.sendTransform(aruco_robot_tf)
            self.tf_broadcaster.sendTransform(aruco_camera_tf)

            rospy.sleep(0.5)



if __name__ == '__main__':
    try:
        transform_publisher = TransformPublisher()
        # transform_publisher.run()
    except rospy.ROSInterruptException:
        pass