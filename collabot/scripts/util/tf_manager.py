from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg import ModelStates,ModelState
import tf
import numpy as np
from gazebo_msgs.srv import SetModelState
import rospy
from util.funcs import quaternion_to_transform_matrix,T_to_xyzrpy
import tf2_ros
from geometry_msgs.msg import TransformStamped

# manage all the tf
# create grasp info


class tf_manager:
    def __init__(self, robot_name_list,obj_list, debug = False):
        self.robot_num = len(robot_name_list)
        self.robot_name_list = robot_name_list
        self.obj_list = obj_list + robot_name_list
        self.tf_listener = tf.TransformListener()

        #obtain the tf info
        #for example: from robot1_agv_base_link to robot1_gripper_finger1_finger_link
        #for each robot, we need to obtain the this kind of tf
        self.ee_pose_dict = dict.fromkeys(robot_name_list,None)
        
        #obtain object and robot pose
        #direct listen to the topic /gazebo/model_states
        self.init_obj_pose = dict.fromkeys(self.obj_list,None)
        self.object_pose_dict = dict.fromkeys(self.obj_list,np.eye(4))


        #grasp info
        self.grasp_info = [] #each element: (robot_name, object name, ee_pose_when_grasp)

        #from ee to camera
        #ee: robot1_gripper_finger1_finger_link
        #camera_link: robot1_d435_color_optical_frame
        # self.gripper_frame = "gripper_finger1_finger_link"
        self.gripper_frame = "tool0"
        self.camera_optical_frame = "d435_color_optical_frame"
        #listen to tf
        self.tf_listener.waitForTransform("robot1_"+self.gripper_frame, "robot1_"+self.camera_optical_frame, rospy.Time(0), rospy.Duration(4.0))
        tf_transform, rotation = self.tf_listener.lookupTransform("robot1_"+self.gripper_frame, "robot1_"+self.camera_optical_frame, rospy.Time(0))
        self.T_ee_2_cam = quaternion_to_transform_matrix(rotation, tf_transform)
        
        #wrist 3 to tool0
        self.tf_listener.waitForTransform("robot1_wrist_3_link", "robot1_"+self.gripper_frame, rospy.Time(0), rospy.Duration(4.0))
        tf_transform, rotation = self.tf_listener.lookupTransform("robot1_wrist_3_link", "robot1_"+self.gripper_frame, rospy.Time(0))
        self.T_wrist3_2_ee = quaternion_to_transform_matrix(rotation, tf_transform)



        # publish world to each robot
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.debug = debug
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        rospy.Subscriber('/gazebo/link_states', ModelStates, self.link_states_callback, queue_size=1)
    
    def get_now_world_robot_pose(self):
        #initial pose for arm is 0
        base_wolrd_pose = [T_to_xyzrpy(self.get_robot_world_pose(robot_name)) for robot_name in self.robot_name_list]
        agv_pose_list = np.zeros((self.robot_num,9)) #only return the agv pose
        for i in range(len(self.robot_name_list)):
            agv_pose_list[i][0] = base_wolrd_pose[i][0]
            agv_pose_list[i][1] = base_wolrd_pose[i][1]
            agv_pose_list[i][2] = base_wolrd_pose[i][-1]
        return agv_pose_list

    def model_states_callback(self, msg):
        # Update the end effector poses for each robot
        #do not update the ee pose here
        # for robot_name in self.robot_name_list:
        #     timenow = rospy.Time.now()
        #     robot_base_name = f"{robot_name}_agv_base_link"
        #     #recomend set to robot1_gripper_finger1_finger_tip_link
        #     ee_name = f"{robot_name}_gripper_finger1_finger_tip_link"
        #     try:
        #         self.tf_listener.waitForTransform(robot_base_name, ee_name, timenow, rospy.Duration(0.5))
        #         tf_transform, rotation = self.tf_listener.lookupTransform(robot_base_name, ee_name, timenow)
        #         ee_pose_trans = quaternion_to_transform_matrix(rotation, tf_transform)
        #         self.ee_pose_dict[robot_name] = ee_pose_trans

        #     except:
        #         pass

        #update object pose from msg
        for obj_name in self.obj_list:
            obj_index = msg.name.index(obj_name)
            obj_pose = msg.pose[obj_index]
            position = [obj_pose.position.x, obj_pose.position.y, obj_pose.position.z]
            orientation = [obj_pose.orientation.x, obj_pose.orientation.y, obj_pose.orientation.z, obj_pose.orientation.w]
            obj_transform = quaternion_to_transform_matrix(orientation, position)
            self.object_pose_dict[obj_name] = obj_transform
            # publish world to robot tf if needed
            # if "robot" in obj_name:
            #     self.pub_world_2_robot(obj_name,position,orientation)
        
        #init obj pose
        for obj_name in self.obj_list:
            if self.init_obj_pose[obj_name] is None:
                self.init_obj_pose[obj_name] = self.object_pose_dict[obj_name]

    def link_states_callback(self, msg):   
        #if one object is added with a grasp pose
        #update the object pose
        # self.set_grasp_obj_pose()
        #update the ee pose
        for robot_name in self.robot_name_list:
            robot_base_name = f"{robot_name}_agv_base_link"
            #recomend set to robot1_gripper_finger1_finger_tip_link
            # ee_name = f'{robot_name}::{robot_name}_{self.gripper_frame}'
            wrist3_name = f'{robot_name}::{robot_name}_wrist_3_link'
            try:
                index = msg.name.index(wrist3_name)
                
                position = [msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z]
                orientation = [msg.pose[index].orientation.x, msg.pose[index].orientation.y, msg.pose[index].orientation.z, msg.pose[index].orientation.w]
                wrist3_pose_trans = quaternion_to_transform_matrix(orientation, position)
                ee_pose_trans = wrist3_pose_trans @ self.T_wrist3_2_ee #world to ee
                
                world_to_robot_trans = self.object_pose_dict[robot_name]
                robot_ee_trans = np.linalg.inv(world_to_robot_trans) @ ee_pose_trans
                self.ee_pose_dict[robot_name] = robot_ee_trans
                
            except ValueError:
                print("Specified link not found in the message.")
        

        
    def pub_world_2_robot(self,robot_name,position,ori):
        #publish the world to robot tf
        #position: [x,y,z]
        #ori: [x,y,z,w]
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "world"
        transform.child_frame_id = f"{robot_name}_agv_base_footprint"
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = ori[0]
        transform.transform.rotation.y = ori[1]
        transform.transform.rotation.z = ori[2]
        transform.transform.rotation.w = ori[3]
        self.tf_broadcaster.sendTransform(transform)

    def get_robot_world_pose(self,robot_name):
        #in simulation
        return self.object_pose_dict[robot_name]
    
    def get_robot_odom(self,robot_name):
        return np.linalg.inv(self.init_obj_pose[robot_name]) @ self.object_pose_dict[robot_name]
    
    def get_robot_ee_pose(self,robot_name):
        #ee pose in robot frame
        return self.ee_pose_dict[robot_name]
    
    def get_obj_pose(self,obj_name):
        return self.object_pose_dict[obj_name]
    
    def get_cam_K_inv(self,robot_name, frame='world'):
        #frame: world or robot_name
        #return the inv of extrinsics of the camera
        if self.T_ee_2_cam is None:
            print("no hand eye calibration")
            return None
        if frame != robot_name:
            return self.object_pose_dict[robot_name] @ self.get_robot_ee_pose(robot_name) @ self.T_ee_2_cam 
        else:
            return self.get_robot_ee_pose(robot_name) @ self.T_ee_2_cam 

    
    def set_obj_pose(self,model_name,position,orientation = (0,0,0,1)):
        rospy.wait_for_service('/gazebo/set_model_state')

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.pose.position.x = position[0]
        model_state.pose.position.y = position[1]
        model_state.pose.position.z = position[2]
        model_state.pose.orientation.x = orientation[0]
        model_state.pose.orientation.y = orientation[1]
        model_state.pose.orientation.z = orientation[2]
        model_state.pose.orientation.w = orientation[3]
        
        resp = set_state(model_state)
      
    def add_grasp(self,robot_name,obj_name):
        #set grasp info
        #get the grasp infomation from object to ee
        T_obj_ee = np.linalg.inv(self.get_obj_pose(obj_name)) @ self.get_obj_pose(robot_name)@ self.get_robot_ee_pose(robot_name)

        self.grasp_info.append((robot_name,obj_name,T_obj_ee))
        
    def remove_grasp(self,robot_name,obj_name):
        #remove grasp info
        for grasp in self.grasp_info:
            if grasp[0] == robot_name and grasp[1] == obj_name:
                self.grasp_info.remove(grasp)

    def direct_set_obj_pose(self,obj_name,position,orientation = (0,0,0,1)):
        #direct set the object pose
        #for example: set the object pose in the world frame
        self.set_obj_pose(obj_name,position,orientation)


    def set_grasp_obj_pose(self):
        have_set_obj = []
        for now_grasp in self.grasp_info:
            robot_name = now_grasp[0]
            obj_name = now_grasp[1]
            T_obj_ee = now_grasp[2]
            if obj_name not in have_set_obj:
                T_world_obj = self.get_obj_pose(robot_name)@ self.get_robot_ee_pose(robot_name) @np.linalg.inv(T_obj_ee)
                pos = T_world_obj[:3,3]
                ori = R.from_matrix(T_world_obj[:3,:3]).as_quat()
                self.set_obj_pose(obj_name,pos,ori)
                have_set_obj.append(obj_name)