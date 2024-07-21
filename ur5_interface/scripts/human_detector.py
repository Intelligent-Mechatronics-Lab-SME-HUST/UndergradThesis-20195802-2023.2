#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
import rospkg
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from hri_msg.msg import HumanJoint
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations


class ROSNode:
    def __init__(self):
        rospy.init_node("YOLO_NODE")
        rospy.loginfo("Starting YOLO_NODE.")
        self.setup_model()
        self.setup_pub_sub()

    def setup_model(self):
        # YOLO_MODEL
        model_path = os.path.join(rospkg.RosPack().get_path("ur5_interface"), "models/yolov8s-pose.pt")
        self.model = YOLO(model_path)
        self.model.fuse()
        # Camera Info
        camera_info = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo)
        camera_info: CameraInfo
        self.camera_matrix = np.array(camera_info.K, dtype=np.float32).reshape(3, 3)
        print(self.camera_matrix)

        # Bridge
        self.bridge = CvBridge()

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def setup_pub_sub(self):
        self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)
        self.img_pub_debug = rospy.Publisher("/img_debug", Image, queue_size=2)
        self.joint_pub = rospy.Publisher("/human_position_topic", HumanJoint, queue_size=10)
        self.marker_human = rospy.Publisher("/human_topic_marker", Marker, queue_size=1)

    def img_callback(self, img_msg: Image):
        img_cv = self.bridge.imgmsg_to_cv2(img_msg)
        img_depth = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)

        results = self.model.track(
            source=img_cv,
            conf=0.7,
            iou=0.8,
            max_det=10,
            device=0,
            classes=0,
            stream=False,
            verbose=False,
        )

        plotted_image = results[0].plot(
            conf=True,
            line_width=1,
            font_size=1,
            font="Arial.ttf",
            labels=True,
            boxes=True,
        )

        if len(results[0].boxes):
            conf = results[0].keypoints.cpu().numpy().conf.tolist()[0]
            points = results[0].keypoints.cpu().numpy().xy.tolist()[0]
            human_joint_list = list(zip(conf, points))
            if human_joint_list[10][0] >= 0.8 or human_joint_list[9][0] >= 0.8:
                self.compute_joint_3D(human_joint_list, img_depth)
        else:
            human_msg = HumanJoint()
            human_msg.exist = False
            self.joint_pub.publish(human_msg)

        self.send_img_debug(plotted_image)

    def compute_joint_3D(self, human_joint_list, img_depth):
        img_depth_cv = self.bridge.imgmsg_to_cv2(img_depth, desired_encoding="passthrough")
        depth_array = np.array(img_depth_cv, dtype=np.float32)
        x_pixel = human_joint_list[10][1][0]
        y_pixel = human_joint_list[10][1][1]
        joint_depth = depth_array[int(y_pixel), int(x_pixel)] / 1000
        x_real = float((x_pixel - self.camera_matrix[0, 2]) / float(self.camera_matrix[0, 0])) * float(joint_depth)
        y_real = float((y_pixel - self.camera_matrix[1, 2]) / float(self.camera_matrix[1, 1])) * float(joint_depth)

        human_mtx = np.asarray([[x_real, y_real, joint_depth,1]]).reshape(4, 1)

        trans = self.tfBuffer.lookup_transform("base_link", "camera_color_optical_frame", rospy.Time())
        trans: TransformStamped
        color_from_base_link = tf.transformations.quaternion_matrix(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]
        )
        color_from_base_link[0, 3] = trans.transform.translation.x
        color_from_base_link[1, 3] = trans.transform.translation.y
        color_from_base_link[2, 3] = trans.transform.translation.z

        human_from_base_link = np.dot(color_from_base_link, human_mtx)

        human_msg = HumanJoint()
        human_msg.position.x = human_from_base_link[0, 0]
        human_msg.position.y = human_from_base_link[1, 0]
        human_msg.position.z = human_from_base_link[2, 0]
        human_msg.exist = True

        self.joint_pub.publish(human_msg)

        human_marker = Marker()
        human_marker.header.frame_id = "camera_color_optical_frame"
        human_marker.type = Marker.SPHERE
        human_marker.action = Marker.ADD
        human_marker.id = 12
        human_marker.pose.position.x = x_real
        human_marker.pose.position.y = y_real
        human_marker.pose.position.z = joint_depth
        human_marker.pose.orientation.w = 1
        human_marker.scale.x = 0.218
        human_marker.scale.y = 0.218
        human_marker.scale.z = 0.218
        human_marker.color.r = 1.0
        human_marker.color.g = 0.0
        human_marker.color.b = 0.0
        human_marker.color.a = 1.0
        human_marker.lifetime = rospy.Duration(2)

        self.marker_human.publish(human_marker)

    def send_img_debug(self, img_cv):
        img_msg = self.bridge.cv2_to_imgmsg(img_cv, encoding="rgb8")
        self.img_pub_debug.publish(img_msg)


if __name__ == "__main__":
    name_node = ROSNode()
    rospy.spin()
