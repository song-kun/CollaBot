import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d

import trimesh
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
from util.funcs import xyzrpy_to_T, T_to_xyzrpy, sample_points


class Jaka_FK():
    def __init__(self,debug = True):
        self.link_name_lst = ['base','shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']
        # wrist3_2_ee_trans = np.array([-0.046, 0.000, 0.108,-1.571, -1.571, 0.000 ]) #xyz rpy
        wrist3_2_ee_trans = np.array([0, 0.000, 0.1,0,0,0])
        self.wrist3_2_ee = xyzrpy_to_T(wrist3_2_ee_trans,xyz_first = True)

        current_path = os.path.dirname(os.path.abspath(__file__))
        if debug:
            # model_path = '~/collabot/src/CollaBot/model_description/jaka_robot/meshes'

            model_path = os.path.join(current_path,'../../../model_description/jaka_robot/meshes')
            self.points = self.load_points(model_path)

        self.dof = 6
        #rpy_xyz
        # self.joint_trans = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0.12015],[-1.5708, 0, 0,0 ,0.14415, 0],[0,0,0,0.36,0,-0.14264],[0,0,0,0.30353, 0, 0.11501],
        #        [-1.5708, 0, 0,0,0.1135,0],[0,0,0,2.7652E-05, -1.3485E-05, -0.021245]])
                
        self.joint_trans = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0.12015],[-1.5708, 0, 0,0 ,0.14415, 0],[0,0,0,0.36,0,-0.14264],[0,0,0,0.30353, 0, 0.11501],
        [-1.5708, 0, 0,0,0.1135,0],[-1.5708, 0, 0,0, 0.107, 0]])

        self.joint_trans_T = []
        for now_ in self.joint_trans:
            self.joint_trans_T.append(xyzrpy_to_T(now_,xyz_first = False))

        #0,z,-z,-z,-z,z,z
        self.rot_axis = [1,1,-1,-1,-1,1,1]
        # self.rot_axis = np.array([[0,0,0],[0,0,1],[0,-1,0],[0,0,-1],[0,0,-1],[0,1,0],[0,0,1]])
        self.T_base_to_link = [np.eye(4,4).astype(float) for i in range(self.dof+1)] #T_base_to_link[i] is the transformation matrix from base to link i
        
        arm = [[] for i in range(7)]
        # arm[0].append([0., 0., 0.02, 0.05]) #open this will lead to mismatch of jacobian matric
        
        arm[1].append([0., 0., 0.02,0.05])

        arm[2].append([0., 0.0, 0, 0.06])
        arm[2].extend([
            # [0.06, 0.0, 0.0, 0.035],
            [0.12, 0.0, 0.0, 0.040],
            # [0.18, 0.0, 0.0, 0.040],
            [0.24, 0.0, 0.0, 0.040],
            # [0.30, 0.0, 0.0, 0.045],
            [0.36, 0.0, 0.0, 0.045],
        ])

        arm[3].append([0., 0.0, 0.03, 0.05])
        arm[3].extend([
            # [0.05, 0.0, 0.0, 0.045],
            [0.10, 0.0, 0.0, 0.045],
            # [0.15, 0.0, 0.0, 0.045],
            [0.20, 0.0, 0.0, 0.045],
            # [0.25, 0.0, 0.0, 0.040],
            [0.30, 0.0, 0.0, 0.040],
        ])

        arm[4].append([0., 0.0, 0, 0.05])
        arm[5].append([0., 0.0, 0, 0.03])
        
        #if consider the collision of joint9, it is very easy to collide with the object when performing grasping
        # arm[6].append([0., 0.0, 0, 0.01])
        #camera collision
        # arm[6].append([0.212,-0.033,0, 0.01])
        self.arm_collision = arm


        self.FK([0,0,0,0,0,0,0])

        
    def load_points(self,model_path):
        points = {}
        for index, link_name in enumerate(self.link_name_lst):
            file_path = os.path.join(model_path, f'{link_name}.STL')
            sampled_points = sample_points(file_path, 10000)
            if index == len(self.link_name_lst) - 1: #last link add ee
                axis_points = np.linspace(0, 0.1, 10).reshape(-1, 1)
                x_axis = np.concatenate([axis_points, np.zeros_like(axis_points), np.zeros_like(axis_points)], axis=1)
                y_axis = np.concatenate([np.zeros_like(axis_points), axis_points, np.zeros_like(axis_points)], axis=1)
                z_axis = np.concatenate([np.zeros_like(axis_points), np.zeros_like(axis_points), axis_points], axis=1)
                ee_points = np.concatenate([x_axis, y_axis, z_axis], axis=0)
                #transform ee points
                ee_points = np.matmul(self.wrist3_2_ee[:3,:3], ee_points.T).T + self.wrist3_2_ee[0:3,3]
                sampled_points = np.concatenate([sampled_points, ee_points], axis=0)

            points[link_name] = sampled_points
            
        return points

    def get_points_pcd(self):
        all_points = self.transform_point_cloud([self.points[link_name] for link_name in self.link_name_lst])
        #vis
        all_points = np.concatenate(all_points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        return pcd
    
    def get_points(self):
        all_points = self.transform_point_cloud([self.points[link_name] for link_name in self.link_name_lst])
        #vis
        all_points = np.concatenate(all_points, axis=0)
        return all_points
    
    def vis_points(self):
        pcd = self.get_points_pcd()
        o3d.visualization.draw_geometries([pcd])
        
    def vis_points_and_collision(self):
        pcd = self.get_points_pcd()
        collision_sphere = self.vis_arm_collision()
        o3d.visualization.draw_geometries([pcd] + collision_sphere)

    def rpy_2_rot(self,r, p, y):
        # 预先计算cos和sin
        cos_y = np.cos(y)
        sin_y = np.sin(y)
        cos_p = np.cos(p)
        sin_p = np.sin(p)
        cos_r = np.cos(r)
        sin_r = np.sin(r)

        # 使用预先计算的值构建旋转矩阵
        now_rot = np.array([
            [
                cos_y * cos_p,
                cos_y * sin_p * sin_r - sin_y * cos_r,
                cos_y * sin_p * cos_r + sin_y * sin_r
            ],
            [
                sin_y * cos_p,
                sin_y * sin_p * sin_r + cos_y * cos_r,
                sin_y * sin_p * cos_r - cos_y * sin_r
            ],
            [
                -sin_p,
                cos_p * sin_r,
                cos_p * cos_r
            ]
        ])
        return now_rot
    
    
    def FK(self,q):
        #q: [7] or [6]
        if len(q) == 6:
            q = [0,q[0],q[1],q[2],q[3],q[4],q[5]]
            
        for i in range(1,self.dof+1):
            now_joint_trans = self.joint_trans_T[i] @ xyzrpy_to_T([0,0,q[i] * self.rot_axis[i],0,0,0],xyz_first = False)
            self.T_base_to_link[i] = np.matmul(self.T_base_to_link[i-1],now_joint_trans)
    
    def transform_point_cloud(self,points):
        #points: [7,N,3]
        new_points = []
        for index, now_points in enumerate(points):
            now_trans = self.T_base_to_link[index]
            tmp = np.matmul(now_trans[:3,:3], now_points.T).T + now_trans[0:3,3]
            new_points.append(tmp)
        return new_points
    
    def transform_point_cloud_index(self,points,index):
        #points: [N,3]
        now_trans = self.T_base_to_link[index]
        tmp = np.matmul(now_trans[:3,:3], points.T).T + now_trans[0:3,3]
        return tmp

    
    def get_arm_collision(self):
        total_center = np.array([]).reshape(-1,3)
        total_radius = np.array([]).reshape(-1)
        for i in range(len(self.arm_collision)):
            now_center_r = np.array(self.arm_collision[i])
            if len(now_center_r) == 0:
                continue
            new_center = self.transform_point_cloud_index(now_center_r[:,0:3],i)
            now_radius = self.arm_collision[i]
            total_center = np.concatenate([total_center,new_center],axis = 0)
            total_radius = np.concatenate([total_radius,np.array(now_radius)[:,3]],axis = 0)
        
        return total_center,total_radius

    def vis_arm_collision(self):
        center,radius = self.get_arm_collision()
        now_sphere = []
        for i in range(len(center)):
            now_sphere.append(self.create_sphere(radius=radius[i],trans = center[i]))
        return now_sphere

    def create_sphere(self,radius=1.0, resolution=20,trans = [0,0,0]):
        trans_mat = np.eye(4)
        trans_mat[0:3, 3] = trans
        # 创建一个球体网格
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sphere.compute_vertex_normals()  # 为了更好的光照效果
        return sphere.transform(trans_mat)

class agv():
    def __init__(self,debug = True):
        self.points = None #N*3
        self.offset_z = -0.2
        current_path = os.path.dirname(os.path.abspath(__file__))
        if debug:
            model_path = os.path.join(current_path,'../../../model_description/agv_description/mesh/collision/')
            # model_path = '~/collabot/src/CollaBot/model_description/agv_description/mesh/collision/'
            self.load_model(model_path)

        self.create_self_collision()
        
    
    def create_self_collision(self):
        base_collision = [[] for i in range(1)]
        delta_x = 0.3
        delta_y = 0.27
        small_value = 0.01
        r= 0.04

        #sparse

        # height = 0.25+self.offset_z
        # base_collision[0].append([delta_x, delta_y, height, r]) #base collision
        # base_collision[0].append([delta_x, -delta_y, height, r])
        # base_collision[0].append([-delta_x, delta_y, height, r])
        # base_collision[0].append([-delta_x, -delta_y, height, r])

        # height = 0.34+self.offset_z
        # base_collision[0].append([delta_x, delta_y, height, r]) #base collision
        # base_collision[0].append([delta_x, -delta_y, height, r])
        # base_collision[0].append([-delta_x, delta_y, height, r])
        # base_collision[0].append([-delta_x, -delta_y, height, r])


        #dense
        for x_ in np.arange(-delta_x,delta_x,2*r+small_value):
            for y_ in np.arange(-delta_y,delta_y,2*r+small_value):
                for z_ in np.arange(0.25+self.offset_z,0.34+self.offset_z,2*r+small_value):
                    base_collision[0].append([x_, y_, z_, r])



        self.base_collision = base_collision

    def load_model(self,model_path):
        base_path = os.path.join(model_path,'agv.stl')
        base_points = sample_points(base_path, 10000)
        wheel_path = os.path.join(model_path,'wheel.stl')
        wheel_points = sample_points(wheel_path, 1000)
        total_points = copy.deepcopy(base_points)
        for offset_x in [0.24,-0.24]:
            for offset_y in [0.275,-0.275]:
                total_points = np.concatenate([total_points,wheel_points + np.array([-offset_x,-offset_y,0.1])],axis = 0)
        total_points[:,2] +=self.offset_z
        self.points = total_points

        rota_mat = R.from_euler('z',1.5708).as_matrix()
        self.points = self.points@rota_mat
    
    def get_agv_pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd
    
    def vis_agv(self):
        pcd = self.get_agv_pcd()
        o3d.visualization.draw_geometries([pcd])
        return pcd
    
    def get_collision(self):
        total_center = np.array([]).reshape(-1,3)
        total_radius = np.array([]).reshape(-1)
        for i in range(len(self.base_collision)):
            now_center_r = np.array(self.base_collision[i])
            if len(now_center_r) == 0:
                continue
            new_center = now_center_r[:,0:3]
            now_radius = now_center_r[:,3]
            total_center = np.concatenate([total_center,new_center],axis = 0)
            total_radius = np.concatenate([total_radius,now_radius],axis = 0)
        
        return total_center,total_radius
    
    def create_sphere(self,radius=1.0, resolution=20,trans = [0,0,0]):
        trans_mat = np.eye(4)
        trans_mat[0:3, 3] = trans
        # 创建一个球体网格
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sphere.compute_vertex_normals()  # 为了更好的光照效果
        return sphere.transform(trans_mat)
    
    def get_points_pcd(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd
    
    def vis_collision(self,vis_coordinate = False):
        center,radius = self.get_collision()
        now_sphere = []
        for i in range(len(center)):
            now_sphere.append(self.create_sphere(radius=radius[i],trans = center[i]))
        pcd = self.get_points_pcd()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])

        if vis_coordinate:
            o3d.visualization.draw_geometries(now_sphere + [pcd,coordinate_frame])
        else:
            o3d.visualization.draw_geometries(now_sphere + [pcd])

class agv_arm():
    def __init__(self,debug = True,device = 'cpu',data_type = torch.float32):
        #robot1_agv_base_link MM
        #robot1_base_link arm
        #
        self.device = 'device'
        self.data_type = data_type
        
        self.agv = agv(debug)
        self.jaka = Jaka_FK(debug)
        self.debug = debug

        self.T_world_2_agv = np.eye(4)  #pose of agv
        self.T_agv_2_arm = np.eye(4)
        self.T_agv_2_arm[0:3,3] = np.array([0.1395,0,0.1965])
        

        # wrist3_2_ee_trans = np.array([-0.046, 0.000, 0.108,-1.571, -1.571, 0.000 ]) #xyz rpy wrist3 to ?
        # wrist3_2_ee_trans = np.array([-0.050, 0.000, 0.104,-0.983, -1.571, 0.000]) #xyz rpy wrist3 to robot1_gripper_finger1_finger_tip_link
        wrist3_2_ee_trans = np.array([0, 0.000, 0.07,0,0,0]) #set to a self defined frame
        self.agv_base_height = 0.207516 #the heigh of base link to ground

        self.wrist3_2_ee = xyzrpy_to_T(wrist3_2_ee_trans,xyz_first = True)

        self.lower_bound = np.array([-100,-100,-6.28,-6.28,-1.48,-3.05,-1.48,-6.28,-6.28])
        self.upper_bound = np.array([100,100,6.28,6.28,4.625,3.05,4.625,6.28,6.28])

        self.dof = 9
        #first 3: x,y,yaw for the agv
        #last 6: 6 joint for the jaka
        self.q = np.zeros(9) 
        self.T_fk = self.FK(self.q)
        self.T_agv_base_2_ee = None

    def move_agv(self,q):
        #q: 9*1
        self.FK(q) 
        self.q = np.array(q)

    def FK(self,q):
        #return: T 9*6 matrix
        self.jaka.FK(q[3:])
        jaka_base_2_link = self.jaka.T_base_to_link[1:] #len = 7

        T_fk = [np.eye(4,4) for i in range(9)]
        T_fk = np.array(T_fk)

        T_fk[0,0,3] = q[0]
        T_fk[1,1,3] = q[1]
        #rotation
        T_fk[2,0:2,0:2] = np.array([[np.cos(q[2]),-np.sin(q[2])],[np.sin(q[2]),np.cos(q[2])]])
        T_fk[1] = T_fk[0] @ T_fk[1]
        T_fk[2] = T_fk[1] @ T_fk[2]

        #calculate for arm
        now_index = 3
        for now_trans in jaka_base_2_link:
            T_fk[now_index] = T_fk[2] @ self.T_agv_2_arm @ now_trans
            now_index += 1
        self.T_fk = T_fk
        self.q = np.array(q)
        return T_fk
    
    def get_ee_T(self,q=None):
        if q is not None:
            T_fk = self.FK(q)
        else:
            T_fk = self.T_fk
        return T_fk[-1] @ self.wrist3_2_ee
    
    def get_colli_points_r_jaco(self,q):
        #return all points and radius
        # if np.linalg.norm(self.q - q) > 1e-3:
        self.FK(q)
        all_points, all_radius = self.get_collision()
        all_jaco = self.get_all_collision_jaco()
        return all_points,all_radius,all_jaco

        
    def get_all_collision_jaco(self):
        #get all points
        # all_points, all_center = self.get_collision()
        #agv collision
        all_jaco = []
        agv_center,agv_radius = self.agv.get_collision() #collision in joint 2
        for now_center in agv_center:
            jaco = self.jacobian_collision(now_center,2)
            all_jaco.append(jaco)
        #joint collision
        #do not consider the collision of wrist3, because this function is only used when grasping
        for i in range(3,9):
            this_joint_center = self.jaka.arm_collision[i-2]
            for now_center in this_joint_center:
                jaco = self.jacobian_collision(now_center[0:3],i)
                all_jaco.append(jaco)
        
        return all_jaco



    def jacobian_collision(self,p_e,link_index):
        #return linear jacobian only
        #p_e from ee to the link
        #link_index: 2-8; 2 for the base, 8 for the wrist 3
        T_fk = self.T_fk
        lin_jac = np.zeros([3, self.dof])
        #x y move
        lin_jac[0, 0] = 1
        lin_jac[1, 1] = 1
        #rot
        for i in range(2, link_index+1):
            if i == 2:
                rot_flag = 1
            else:
                rot_flag = self.jaka.rot_axis[i-2]
            pose = T_fk[i]
            p_i, z_i = pose[:3, 3], rot_flag * pose[:3, 2]
            lin_jac[:, i] = np.cross(z_i, p_e - p_i)
        return lin_jac

        

    def jacobian(self,q):
        #calculate the jacobian matrix for current agv arm
        #return T 9*6
        T_fk = self.FK(q)
        T_agv_base_2_ee = T_fk[-1] @ self.wrist3_2_ee
        self.T_agv_base_2_ee = T_agv_base_2_ee

        p_e = T_agv_base_2_ee[:3,3] # the end effector position
        lin_jac, ang_jac = np.zeros([3, self.dof]), np.zeros([3, self.dof])
        #x y move
        lin_jac[0, 0] = 1
        lin_jac[1, 1] = 1
        #rot
        for i in range(2, self.dof):
            if i == 2:
                rot_flag = 1
            else:
                rot_flag = self.jaka.rot_axis[i-2]
            pose = T_fk[i]
            p_i, z_i = pose[:3, 3], rot_flag * pose[:3, 2]
            lin_jac[:, i] = np.cross(z_i, p_e - p_i)
            ang_jac[:, i] = z_i
        jaco = np.vstack((lin_jac, ang_jac))
        return jaco
    
    def next_ee_pose_using_jacobian(self,jaco,dq):
        #jaco: 6*9
        #dq: 9*1
        delta = jaco @ dq
        new_rot = R.from_rotvec(delta[3:]).as_matrix() @ self.T_agv_base_2_ee[:3,:3]
        new_trans = self.T_agv_base_2_ee[:3,3] + delta[:3]
        T_final = np.eye(4)
        T_final[:3,:3] = new_rot
        T_final[:3,3] = new_trans
        return T_final


    def get_collision(self):
        self.agv_center,self.agv_radius = self.agv.get_collision()
        self.jaka_center,self.jaka_radius = self.jaka.get_arm_collision()

        self.jaka_center = self.arm_points_2_agv(self.jaka_center)

        #transform all points due to odom
        self.agv_center = self.points_2_world(self.agv_center)
        self.jaka_center = self.points_2_world(self.jaka_center)
        
        return [self.agv_center,self.jaka_center],[self.agv_radius,self.jaka_radius]
    
    def points_2_world(self,points):
        #from agv_base_link to world
        self.T_world_2_agv = self.T_fk[2]
        return np.matmul(self.T_world_2_agv[:3,:3], points.T).T + self.T_world_2_agv[0:3,3]
    
    def arm_points_2_agv(self,points):
        #points: N *3
        return np.matmul(self.T_agv_2_arm[:3,:3], points.T).T + self.T_agv_2_arm[0:3,3]

    def create_sphere(self,radius=1.0, resolution=20,trans = [0,0,0]):
        trans_mat = np.eye(4)
        trans_mat[0:3, 3] = trans
        # 创建一个球体网格
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sphere.compute_vertex_normals()  # 为了更好的光照效果
        return sphere.transform(trans_mat)
    
    def vis_collision(self,vis_coordinate = False,show = True,show_obj_list = []):
        if show:
            vis_data = []
            for show_obj in show_obj_list:
                vis_data.append(show_obj)
            if vis_coordinate:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                vis_data.append(axis)

            #vis collision ball
            collision_center,collision_r = self.get_collision()

            for now_center,now_radius in zip(collision_center,collision_r):
                for center_, radius_ in zip(now_center, now_radius):
                    vis_data.append(self.create_sphere(radius=radius_, trans=center_))
            if self.debug:
                #vis agv
                pcd = self.agv.get_points_pcd()
                #transform to world
                agv_points = self.points_2_world(np.array(pcd.points))
                pcd.points = o3d.utility.Vector3dVector(agv_points)
                vis_data.append(pcd)

                #vis jaka
                pcd = self.jaka.get_points_pcd()
                #transform to agv
                arm_points = self.points_2_world(self.arm_points_2_agv(np.array(pcd.points)))
                pcd.points = o3d.utility.Vector3dVector(arm_points)
                vis_data.append(pcd)
                
                if vis_coordinate:
                    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
                    vis_data.append(coordinate_frame)
        
            o3d.visualization.draw_geometries(vis_data)
        else:
            agv_points = self.points_2_world(copy.deepcopy(self.agv.points))
            arm_points = copy.deepcopy(self.jaka.get_points())
            arm_points = self.points_2_world(self.arm_points_2_agv(arm_points))
            return agv_points,arm_points
    
    
        
        
    def animation(self,qs,ts):
        # visualize motion planning
        agv_points,arm_points = self.vis_collision(show=False)
        
        agv_pcd = o3d.geometry.PointCloud()
        arm_pcd = o3d.geometry.PointCloud()
        agv_pcd.points = o3d.utility.Vector3dVector(agv_points)
        arm_pcd.points = o3d.utility.Vector3dVector(arm_points)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(agv_pcd)
        vis.add_geometry(arm_pcd)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
        vis.add_geometry(coordinate_frame)

        # 设置相机的参数
        view_ctl = vis.get_view_control()
        view_ctl.set_front([-4, 0., 4])  # 相机pose
        view_ctl.set_lookat([1, 0, 0])       # 相机焦点
        view_ctl.set_up([0, 0, 1])           # 相机上方向
        view_ctl.set_zoom(2)               # 缩放级别

        for q,t in zip(qs,ts):
            self.move_agv(q)
            agv_points,arm_points = self.vis_collision(show=False)

            agv_pcd.points = o3d.utility.Vector3dVector(agv_points)
            arm_pcd.points = o3d.utility.Vector3dVector(arm_points)

            vis.update_geometry(agv_pcd)
            vis.update_geometry(arm_pcd)
                        
            vis.poll_events()
            vis.update_renderer()

            time.sleep(0.05)
        vis.destroy_window()

    





if __name__ == "__main__":
    jaka_fk = Jaka_FK()
    q = [0, 0, 0, 0, 0, 0, 0]
    jaka_fk.FK(q)
    print(jaka_fk.T_base_to_link[-1])

    # qs = np.array([[0, 0, 1.5, -1.5, 0, 1, 0],[0,0,0,0,0,0,0]])
    # result = jaka_fk.FK_batch(qs)

    jaka_fk.vis_points_and_collision()

    now_agv = agv()
    now_agv.vis_collision(vis_coordinate=True)

    now_agv_arm = agv_arm()
    now_agv_arm.vis_collision()
    #example for move the car
    # now_agv_arm.move_agv_arm(q = [0, 0, 1, 0, 0, 1, 0],agv_move = [0.1,0.1,0.1])
    # now_agv_arm.vis_collision(vis_coordinate=True)

    num_pose = 100
    qs = np.zeros((num_pose,9))
    qs[:,2] = np.linspace(0,1,num_pose)
    ts = [1 for i in range(num_pose)]
    # now_agv_arm.animation(qs,ts)

    #example for jaco
    q = np.array([0,0,0, 1, 0, 0, 0, 0, 1])
    dq = np.array([0,0,0,    0, 1e-3,0,0,0,1e-3])
    T_base_2_ee = now_agv_arm.get_ee_T(q + dq)
    q_ee = T_to_xyzrpy(T_base_2_ee)

    jaco = now_agv_arm.jacobian(q)
    T_final = now_agv_arm.next_ee_pose_using_jacobian(jaco,dq)
    q_ee_2 = T_to_xyzrpy(T_final)
    print("error final \n",q_ee - q_ee_2)

    #example for collision
    all_points,all_radius,all_jaco = now_agv_arm.get_colli_points_r_jaco(q)
    print(len(all_jaco))

