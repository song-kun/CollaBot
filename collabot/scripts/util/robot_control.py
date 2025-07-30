import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Empty


class RobotControl:
    def __init__(self, robot_name_list):
        self.robot_num = len(robot_name_list)
        self.robot_name_list = robot_name_list
        self.base_publishers = {}
        self.arm_publishers = {}
        self.gripper_publishers = {}
        

        # 初始化每个机器人的发布器
        for robot_name in robot_name_list:
            base_topic = f"/{robot_name}/base_control"
            arm_topic = f"/{robot_name}/arm_control"
            gripper_topic = f"/{robot_name}/gripper_control"
            
            self.base_publishers[robot_name] = rospy.Publisher(base_topic, Twist, queue_size=10)
            self.arm_publishers[robot_name] = rospy.Publisher(arm_topic, Float64MultiArray, queue_size=10)
            self.gripper_publishers[robot_name] = rospy.Publisher(gripper_topic, Float64, queue_size=10)

    def base_v(self, robot_name, v_list):
        # v_list: [x, y, w]
        if robot_name in self.base_publishers:
            twist_msg = Twist()
            twist_msg.linear.x = v_list[0]
            twist_msg.linear.y = v_list[1]
            twist_msg.angular.z = v_list[2]
            self.base_publishers[robot_name].publish(twist_msg)
        else:
            rospy.logwarn(f"Robot name {robot_name} not found in base publishers.")

    def arm_pose(self, robot_name, pose_list):
        # pose_list: 6 axis angles
        if robot_name in self.arm_publishers:
            arm_msg = Float64MultiArray()
            arm_msg.data = pose_list
            self.arm_publishers[robot_name].publish(arm_msg)
        else:
            rospy.logwarn(f"Robot name {robot_name} not found in arm publishers.")

    def gripper_pose(self, robot_name, gripper_pose):
        # gripper_pose: float, 0-1
        if robot_name in self.gripper_publishers:
            gripper_msg = Float64()
            gripper_msg.data = gripper_pose
            self.gripper_publishers[robot_name].publish(gripper_msg)
        else:
            rospy.logwarn(f"Robot name {robot_name} not found in gripper publishers.")
    
    def reset_scene(self):
        for robot_name in self.robot_name_list:
            self.base_v(robot_name, [0, 0, 0])
            self.arm_pose(robot_name, [0, 0, 0, 0, 0, 0])
            self.gripper_pose(robot_name, 0)
        #call service to reset the scene
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        rospy.loginfo("Gazebo world has been successfully reset.")
