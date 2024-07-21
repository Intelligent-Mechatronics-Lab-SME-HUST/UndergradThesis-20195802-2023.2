#! /usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sys
import moveit_msgs.msg
import tf.transformations
import numpy as np
import tf

import os
import pickle

from hri_msg.msg import HumanJoint
from moveit_msgs.msg import RobotTrajectory


def replan_RRT_callback(msg:HumanJoint):
    print("REPLAN")
    scene.remove_world_object("human")
    human = geometry_msgs.msg.PoseStamped()
    human.header.frame_id = group.get_planning_frame()
    human.pose.orientation.w = 1.0
    human.pose.position.x = msg.position.x
    human.pose.position.y = msg.position.y
    human.pose.position.z = msg.position.z
    scene.add_sphere("human", human, radius = 0.0545)

    group.set_pose_target(pose_goal)
    success = False
    while not success:
        success, plan, *other = group.plan(pose_goal)

    pub_traj.publish(plan)


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_interface_tutorial", anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    pub_traj = rospy.Publisher("ref_traj", moveit_msgs.msg.RobotTrajectory, queue_size=10)
    replan_sub = rospy.Subscriber("replan_RRT", HumanJoint, replan_RRT_callback)

    GROUP_NAME = "manipulator"
    group = moveit_commander.MoveGroupCommander(GROUP_NAME)
    group.set_planner_id("BiRRT")
    group.set_planning_time(5)

    print(f"========== Planner: {group.get_planner_id()}")

    #!- Create Planning Scence -!#
    scene.clear()

    plane = geometry_msgs.msg.PoseStamped()
    plane.header.frame_id = group.get_planning_frame()
    plane.pose.orientation.w = 1.0
    plane.pose.position.z = -0.5 / 2
    plane.pose.position.x = 0.9 / 2 - 0.08
    plane.pose.position.y = 1.3 / 2 - 0.34
    plane_name = "plane"
    scene.add_box(plane_name, plane, size=(0.90, 1.3, 0.5))

    wall = geometry_msgs.msg.PoseStamped()
    wall.header.frame_id = group.get_planning_frame()
    wall.pose.orientation.w = 1.0
    wall.pose.position.z = 1 / 2
    wall.pose.position.x = 0.9 / 2 - 0.08
    wall.pose.position.y = -0.34
    wall_name = "wall"
    scene.add_box(wall_name, wall, size=(0.9, 0.01, 1))

    wall.pose.position.y = 1.3 - 0.34
    wall_name = "wall2"
    scene.add_box(wall_name, wall, size=(0.9, 0.01, 1))

    #!- Add A Box -!#
    box_1 = geometry_msgs.msg.PoseStamped()
    box_1.header.frame_id = group.get_planning_frame()
    box_1.pose.orientation.w = 1.0
    box_1.pose.position.z = 0.01 / 2
    box_1.pose.position.x = 0.45
    box_1.pose.position.y = -0.05
    box_1_name = "box1"
    scene.add_box(box_1_name, box_1, size=(0.28, 0.41, 0.01))

    box_2 = geometry_msgs.msg.PoseStamped()
    box_2.header.frame_id = group.get_planning_frame()
    box_2.pose.orientation.w = 1.0
    box_2.pose.position.x = 0.45
    box_2.pose.position.y = -0.05 - 0.41 / 2
    box_2.pose.position.z = 0.11
    box_2_name = "box2"
    scene.add_box(box_2_name, box_2, size=(0.28, 0.01, 0.22))

    box_2.pose.position.y = -0.05 + 0.41 / 2
    box_2_name = "box3"
    scene.add_box(box_2_name, box_2, size=(0.28, 0.01, 0.22))

    box_4 = geometry_msgs.msg.PoseStamped()
    box_4.header.frame_id = group.get_planning_frame()
    box_4.pose.orientation.w = 1.0
    box_4.pose.position.z = 0.11
    box_4.pose.position.x = 0.45 + 0.28 / 2
    box_4.pose.position.y = -0.05
    box_4_name = "box4"
    scene.add_box(box_4_name, box_4, size=(0.01, 0.41, 0.22))

    box_4.pose.position.x = 0.45 - 0.28 / 2
    box_4_name = "box5"
    scene.add_box(box_4_name, box_4, size=(0.01, 0.41, 0.22))

    #!- Camera Base -!#
    camera_ = geometry_msgs.msg.PoseStamped()
    camera_size = (0.91, 0.5, 0.4)
    camera_.header.frame_id = "plane"
    camera_.pose.orientation.w = 1.0
    camera_.pose.position.x = 0
    camera_.pose.position.y = 0.65 - 0.3 / 2 - 0.1
    # camera_.pose.position.z = 0.85 + camera_size[2]/2
    camera_.pose.position.z = 1 + camera_size[2] / 2
    camera_name = "camera_cage"
    scene.add_box(camera_name, camera_, size=camera_size)

    rospy.sleep(2)
    # !- Go Home Pose -!#
    group.set_named_target("home")
    group.go(wait=True)

    #!- Set Start Pose -!#
    # quaternion = tf.transformations.quaternion_from_euler(-2.918, 0.123, -0.715)
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.position.x = 0.1
    # pose_goal.position.y = 0.601
    # pose_goal.position.z = 0.441
    # pose_goal.orientation.x = quaternion[0]
    # pose_goal.orientation.y = quaternion[1]
    # pose_goal.orientation.z = quaternion[2]
    # pose_goal.orientation.w = quaternion[3]
    # group.set_pose_target(pose_goal)
    # group.go(wait=True)

    joint_target = [0.954, -0.893, 0.901, -1.567, -1.559, 0.0]
    group.set_joint_value_target(joint_target)
    group.go(wait=True)

    # #!- Set Goal Pose -!#
    quaternion = tf.transformations.quaternion_from_euler(np.pi, 0, -np.pi / 2)
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.45
    pose_goal.position.y = -0.05
    pose_goal.position.z = 0.4
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    group.set_pose_target(pose_goal)

    # success, plan, *other = group.plan(pose_goal)

    # saved_plan = group.plan(pose_goal)

    file_path = os.path.join(os.path.expanduser("~"), "saved_trajectories", "plan.pickle")
    
    #!- Publish Reference Traj -!#
    usr = input("Confirm Plan: ")
    # group.execute(plan)

    # while(usr == "r"):
    #     saved_plan = group.plan(pose_goal)
        # success, plan, *other = group.plan(pose_goal)
        # usr = input("Confirm Plan: ")

    # with open(file_path, 'wb') as fp:
    #     pickle.dump(saved_plan, fp)

    with open(file_path, 'rb') as file_open:
        plan = pickle.load(file_open)[1]
        # input()
        # rospy.sleep(1.5)
        pub_traj.publish(plan)
        # rospy.sleep(2)
        # group.execute(plan)
        # print(plan)

    # rospy.spin()