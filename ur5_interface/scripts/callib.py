#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations
import yaml

#!- Camera Matrix -!#
# MTX_K = np.asarray(
#     [
#         [607.2698364257812, 0.0, 316.8871154785156],
#         [0.0, 607.2440185546875, 245.54869079589844],
#         [0.0, 0.0, 1.0],
#     ]
# )
MTX_D = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])


def save_transform_to_yaml(transform_stamped, file_path):
    """
    Save a geometry_msgs/TransformStamped message to a YAML file.

    Args:
        transform_stamped (geometry_msgs/TransformStamped): The transform stamped message to save.
        file_path (str): The path to the YAML file to save the transform to.
    """
    data = {
        "header": {
            "stamp": {"secs": transform_stamped.header.stamp.secs, "nsecs": transform_stamped.header.stamp.nsecs},
            "frame_id": transform_stamped.header.frame_id,
        },
        "child_frame_id": transform_stamped.child_frame_id,
        "transform": {
            "translation": {
                "x": transform_stamped.transform.translation.x,
                "y": transform_stamped.transform.translation.y,
                "z": transform_stamped.transform.translation.z,
            },
            "rotation": {
                "x": transform_stamped.transform.rotation.x,
                "y": transform_stamped.transform.rotation.y,
                "z": transform_stamped.transform.rotation.z,
                "w": transform_stamped.transform.rotation.w,
            },
        },
    }
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


#!- Main -!#
if __name__ == "__main__":
    rospy.init_node("Aruco_node")
    rospy.loginfo("Starting Aruco_node.")

    bridge = CvBridge()

    # TF
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    transform = TransformStamped()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # transform.header.frame_id = "camera_color_optical_frame"
    # transform.child_frame_id = "aruco"

    transform.header.frame_id = "camera_link"
    transform.child_frame_id = "aruco"

    camera_info = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo)
    camera_info: CameraInfo
    MTX_K = np.array(camera_info.K, dtype=np.float32).reshape(3, 3)

    while not rospy.is_shutdown():
        print("*" * 10)
        input()

        img_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        img_cv = bridge.imgmsg_to_cv2(img_msg)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(img_cv)

        if len(corners) > 0:
            print("Detected")
            for i in range(0, len(ids)):
                rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, MTX_K, MTX_D)
                image_new = cv2.aruco.drawDetectedMarkers(img_cv.copy(), corners, ids=ids, borderColor=(0, 255, 0))
                image_new = cv2.cvtColor(image_new, cv2.COLOR_RGB2BGR)
                image_new = cv2.drawFrameAxes(image_new, MTX_K, MTX_D, rvecs, tvecs, length=0.1)

                cv2.imwrite("Calib_ARUCO.png", image_new)

                t_x, t_y, t_z = tvecs.tolist()[0][0]
                # roll, pitch, yaw = rvecs.tolist()[0][0]
                aruco_from_color = np.eye(4)
                aruco_from_color[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs.tolist()[0]))[0]
                aruco_from_color[0,3] = t_x
                aruco_from_color[1,3] = t_y
                aruco_from_color[2,3] = t_z
                trans = tfBuffer.lookup_transform("camera_link", "camera_color_optical_frame", rospy.Time(2))
                trans: TransformStamped

                quaternion = [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ]
                color_from_camera_link = tf.transformations.quaternion_matrix(quaternion)

                color_from_camera_link[0,3] = trans.transform.translation.x
                color_from_camera_link[1,3] = trans.transform.translation.y
                color_from_camera_link[2,3] = trans.transform.translation.z

                aruco_from_camera_link = np.dot(color_from_camera_link,aruco_from_color)
                
                transform.header.stamp = rospy.Time.now()
                quaternion = tf.transformations.quaternion_from_matrix(aruco_from_camera_link)
                transform.transform.translation.x = aruco_from_camera_link[0,3]
                transform.transform.translation.y = aruco_from_camera_link[1,3]
                transform.transform.translation.z = aruco_from_camera_link[2,3]
                transform.transform.rotation.x = float(quaternion[0])
                transform.transform.rotation.y = float(quaternion[1])
                transform.transform.rotation.z = float(quaternion[2])
                transform.transform.rotation.w = float(quaternion[3])

                tf_broadcaster.sendTransform(transform)

                np.savetxt("/home/drx/catkin_ws/src/ur5_interface/data/aruco_camera.txt", aruco_from_camera_link)

                
