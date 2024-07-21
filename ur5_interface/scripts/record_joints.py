#! /usr/bin/env python3

import rosbag
import rospy
from sensor_msgs.msg import JointState
from hri_msg.msg import HumanJoint


def record_topics():
    # Initialize the ROS node
    rospy.init_node("REC", anonymous=True)

    # Get the topic names to record from the parameter server
    topic1_name = "/ur5/joint_states"
    topic2_name = "/human_position_topic"

    # Get the bag file name from the parameter server
    bag_file = "/home/drx/catkin_ws/src/ur5_interface/data/0005_4000.bag"

    # Open the bag file for writing
    bag = rosbag.Bag(bag_file, "w")

    def callback1(msg):
        bag.write("/joint_states", msg)

    # def callback2(msg):
        # bag.write(topic2_name, msg)

    rospy.Subscriber(topic1_name, JointState, callback1)
    # rospy.Subscriber(topic2_name, HumanJoint, callback2)

    # Keep the node running until interrupted
    rospy.spin()

    # Close the bag file
    bag.close()
    print(f"Saved bag file: {bag_file}")


if __name__ == "__main__":
    try:
        record_topics()
    except rospy.ROSInterruptException:
        pass
