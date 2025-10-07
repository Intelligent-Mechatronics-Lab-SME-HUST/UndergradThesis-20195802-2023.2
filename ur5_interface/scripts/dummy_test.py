#!/usr/bin/env python3
import rospy
from ur5_interface.msg import HumanJoint
from visualization_msgs.msg import Marker

if __name__ == "__main__":
    rospy.init_node("Dummy_node")
    rospy.loginfo("Starting Dummy_node.")

    dummy_pub = rospy.Publisher("human_position_topic", HumanJoint, queue_size=1)
    marker_dummy = rospy.Publisher("/human_topic_marker", Marker, queue_size=1)

    dummy_msg = HumanJoint()
    dummy_msg.exist = True
    
    # --- Replan in plan_3.pickle ---
    # dummy_msg.position.x = 0.5
    # dummy_msg.position.y = 0.45
    # dummy_msg.position.z = 0.3
    # ----------------------------------

    # --- plan_3.pickle ---
    dummy_msg.position.x = 0.55
    dummy_msg.position.y = 0.4
    dummy_msg.position.z = 0.24
    # ----------------------------------

    dummy_marker = Marker()
    dummy_marker.header.frame_id = "base_link"
    dummy_marker.type = Marker.SPHERE
    dummy_marker.action = Marker.ADD
    dummy_marker.id = 1
    dummy_marker.pose.position.x = dummy_msg.position.x
    dummy_marker.pose.position.y = dummy_msg.position.y
    dummy_marker.pose.position.z = dummy_msg.position.z
    dummy_marker.pose.orientation.w = 1
    dummy_marker.scale.x = 0.109
    dummy_marker.scale.y = 0.109
    dummy_marker.scale.z = 0.109
    dummy_marker.color.r = 0.0
    dummy_marker.color.g = 0.0
    dummy_marker.color.b = 1.0
    dummy_marker.color.a = 0.5
    dummy_marker.lifetime = rospy.Duration(2)

    while not rospy.is_shutdown():
        dummy_pub.publish(dummy_msg)
        if dummy_msg.exist:
            marker_dummy.publish(dummy_marker)
