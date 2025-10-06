#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

void add_box(moveit::planning_interface::PlanningSceneInterface &planning_scene_interface, moveit::planning_interface::MoveGroupInterface &move_group_interface)
{
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_interface.getPlanningFrame();
    collision_object.id = "box1";
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 0.15;
    primitive.dimensions[primitive.BOX_Y] = 0.05;
    primitive.dimensions[primitive.BOX_Z] = 0.5;
    geometry_msgs::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.6;
    box_pose.position.y = 0;
    box_pose.position.z = 0.2;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    std::vector<moveit_msgs::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);
    planning_scene_interface.addCollisionObjects(collision_objects);
}
void add_box2(moveit::planning_interface::PlanningSceneInterface &planning_scene_interface, moveit::planning_interface::MoveGroupInterface &move_group_interface)
{
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_interface.getPlanningFrame();
    collision_object.id = "box1";
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 0.15;
    primitive.dimensions[primitive.BOX_Y] = 0.05;
    primitive.dimensions[primitive.BOX_Z] = 0.5;
    geometry_msgs::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = 0.488;
    box_pose.position.y = 0.3;
    box_pose.position.z = 0.2;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    std::vector<moveit_msgs::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);
    planning_scene_interface.addCollisionObjects(collision_objects);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ur5_interface");
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(0);
    spinner.start();

    static const std::string PLANNING_GROUP = "arm";

    moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    planning_scene_interface.clear();

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    const moveit::core::JointModelGroup *joint_model_group =
        move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("base_link");

    visual_tools.deleteAllMarkers();

    move_group_interface.allowReplanning(true);
    move_group_interface.setReplanAttempts(10);
    move_group_interface.setReplanDelay(0.01);

    // while (1)
    // {
    geometry_msgs::Pose pose_1;
    pose_1.position.x = 0.488;
    pose_1.position.y = -0.25;
    pose_1.position.z = 0.48;

    tf2::Quaternion quat;
    quat.setRPY(180 * M_PI / 180, 0 * M_PI / 180, -90 * M_PI / 180);
    pose_1.orientation.w = quat.w();
    pose_1.orientation.x = quat.x();
    pose_1.orientation.y = quat.y();
    pose_1.orientation.z = quat.z();

    move_group_interface.setPoseTarget(pose_1);
    move_group_interface.move();


    geometry_msgs::Pose pose_2;
    pose_2.position.x = 0.488;
    pose_2.position.y = 0.4;
    pose_2.position.z = 0.2;

    pose_2.orientation.w = quat.w();
    pose_2.orientation.x = quat.x();
    pose_2.orientation.y = quat.y();
    pose_2.orientation.z = quat.z();

    move_group_interface.setPoseTarget(pose_2);
    move_group_interface.plan(my_plan);
    visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    visual_tools.trigger();

    move_group_interface.asyncMove();
    sleep(1);
    add_box(planning_scene_interface,move_group_interface);
    // sleep(1);
    // add_box2(planning_scene_interface,move_group_interface);


    //     sleep(2);
    // }

    ros::waitForShutdown();
    return 0;
}