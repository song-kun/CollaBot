from scipy.spatial.distance import cdist
import torch
import numpy as np

import open3d as o3d
from scipy.spatial.transform import Rotation as R

def generate_T_with_fixed_z(z_axis, N):
    # Generate N evenly spaced rotation angles from 0 to 360 degrees (excluding 360°).
    angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
    z_axis = z_axis / np.linalg.norm(z_axis)  
    original_mat = R.from_euler('z',angles).as_matrix()

    # rotation mat
    dir = np.cross(np.array([0,0,1]), z_axis)
    if np.linalg.norm(dir) < 1e-6:
        dir = np.array([1,0,0])
    else:
        dir = dir / np.linalg.norm(dir)
    angle = np.arccos(np.dot(np.array([0,0,1]), z_axis))
    rot_mat = R.from_rotvec(dir * angle).as_matrix()  

    result = np.einsum('jk,ikn->ijn',rot_mat,original_mat)
    return angles,result



class grasper_model:
    def __init__(self,n_point = 5,d_type = torch.float32,device = 'cpu'):
        #n_point: how many point to fit the gripper model
        self.device = device
        self.d_type = d_type

        self.thickness = 0.01
        self.d = 0.1 # the length that the gripper can open

        self.length = 0.08
        self.width = 0.03

        self.end_length = 0.1
        self.end_width = 0.08
        self.n_point = n_point
        
        self.points,self.radius = self.create_balls()
        self.torch_points = torch.from_numpy(self.points).to(self.device).type(d_type)
        self.torch_radius = torch.from_numpy(self.radius).to(self.device).type(d_type)
        # self.vis_all_points()

        self.dis_th = 0.01
        

    def create_balls(self):
        #the gripper is divided into 3 parts
        #1. the upper part
        center1 = [-self.length/2, 0, self.d/2 + self.thickness/2]
        center2 = [-self.length/2, 0, -self.d/2 - self.thickness/2]
        center3 = [-self.length - self.end_length/2, 0, 0]
        l_x1,l_y1,l_z1 = self.length, self.width, self.thickness
        l_x2,l_y2,l_z2 = self.length, self.width, self.thickness
        l_x3,l_y3,l_z3 = self.end_length, self.end_width, 2*self.thickness+self.d

        points1, radius1 = self.create_ball_in_rect(l_x1, l_y1, l_z1)
        points2, radius2 = self.create_ball_in_rect(l_x2, l_y2, l_z2)
        points3, radius3 = self.create_ball_in_rect(l_x3, l_y3, l_z3)

        points1 = points1  + np.array(center1)
        points2 = points2  + np.array(center2)
        points3 = points3  + np.array(center3)

        #merge all points
        points = np.vstack((points1, points2, points3))
        radius = np.concatenate((radius1, radius2, radius3))
        return points,radius
    
    def create_ball_in_rect(self, h, w, l):
        #center: 0,0,0
        #return points and radius
        radius = min(h, w, l) / self.n_point/2
        points = []
        #create mesh
        x = np.arange(-h/2+radius, h/2 -radius, 2 * radius)
        y = np.arange(-w/2+radius, w/2 -radius, 2 * radius)
        z = np.arange(-l/2+radius, l/2 -radius, 2 * radius)

        #create mesh
        point_x, point_y, point_z = np.meshgrid(x, y, z)
        point_x = point_x.reshape(-1)
        point_y = point_y.reshape(-1)
        point_z = point_z.reshape(-1)
        points = np.vstack((point_x, point_y, point_z)).T
        all_rads = radius * np.ones((points.shape[0],))
        return points, all_rads
    
    def create_sphere(self,radius=1.0, resolution=20,trans = [0,0,0]):
        trans_mat = np.eye(4)
        trans_mat[0:3, 3] = trans
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        return sphere.transform(trans_mat)
    
    def vis_all_points(self):
        #axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        # balls = []
        # for i in range(self.points.shape[0]):
        #     ball = self.create_sphere(radius=self.radius[i],resolution = 2,trans=self.points[i])
        #     balls.append(ball)

        # print("total balls: ", self.points.shape[0])
        # o3d.visualization.draw_geometries(balls)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        o3d.visualization.draw_geometries([pcd,axis])
    
    def check_collision(self,T_world_2_gripper,obj_points):
        #check if the gripper collide with the object
        #T_world_2_gripper: the transformation from world frame to gripper frame
        #obj_points: the object points in the world frame
        #return: True if collide, False if not collide
        #get the gripper points in the world frame
        gripper_points = (T_world_2_gripper[0:3,0:3] @ self.points.T).T + T_world_2_gripper[0:3,3]
        dis_mat = cdist(obj_points,gripper_points)
        

        min_dis = np.min(dis_mat - self.radius, axis=1) 

        if np.min(min_dis) < self.dis_th:
            return True

        return False

    def check_collision_torch(self,T_world_2_gripper,obj_points):
        #check if the gripper collide with the object
        #T_world_2_gripper: the transformation from world frame to gripper frame
        #obj_points: the object points in the world frame
        #return: True if collide, False if not collide
        #get the gripper points in the world frame
        gripper_points = (T_world_2_gripper[0:3,0:3] @ self.torch_points.T).T + T_world_2_gripper[0:3,3]
        dis_mat = torch.cdist(obj_points,gripper_points)

        min_dis = torch.min(dis_mat - self.torch_radius, axis=1).values

        if torch.min(min_dis) < self.dis_th:
            return True

        return False
    
    
    def check_collision_batch(self, T_list, obj_points):
        with torch.no_grad():
            # Convert list of transformation matrices to a tensor
            T_batch = torch.stack(T_list)  # Shape: [batch_size, 4, 4]
            T_batch = T_batch.to(self.device)

            # Extract rotation matrices and translation vectors from the batch
            rot_matrices = T_batch[:, :3, :3]  # Shape: [batch_size, 3, 3]
            translations = T_batch[:, :3, 3]  # Shape: [batch_size, 3]

            # Transform all gripper points for all transformations in the batch

            gripper_points_batch = torch.einsum('bij,jk->bik', rot_matrices, self.torch_points.T).transpose(1,2) + translations.unsqueeze(1)

            collision_results = []
            for gripper_points in gripper_points_batch:
                dis_mat = torch.cdist(obj_points, gripper_points)
                min_dis = torch.min(dis_mat - self.torch_radius, dim=1).values
                collision_results.append(torch.min(min_dis) < self.dis_th)
            #to tensor
            collision_results = torch.tensor(collision_results).to(self.device)

        return collision_results

    def generate_potential_grasp_points(self,points,normal_vectors,collision_points):
        #generate the potential grasp points based on collision check
        #points: the object points
        #normal_vectors: the normal vectors of the object points
        #collision_points: the points that the gripper should not collide with
        #return: the N*3 grasp points
        N_per_point = 40
        suitable_ratio = 0.3

        T_list= []
        point_list = []
        angle_list = []
        #generate the possible grasp pose
        for now_point,now_normal in zip(points,normal_vectors):
            #condition about the working space

            angles,rot_mats = generate_T_with_fixed_z(now_normal, N_per_point)

            for angle_,rot_ in zip(angles,rot_mats):
                T_world_2_gripper = np.eye(4)
                T_world_2_gripper[0:3,0:3] = rot_
                T_world_2_gripper[0:3,3] = now_point
                T_list.append(torch.from_numpy(T_world_2_gripper).type(self.d_type))
                point_list.append(now_point)
                angle_list.append(angle_)
        
        collision_points = torch.from_numpy(collision_points).to(self.device) #all points for the object
        collision_points = collision_points.type(self.d_type)

        result = self.check_collision_batch(T_list, collision_points)
        result = result.cpu().numpy()
        collision_points = collision_points.cpu()

        #check the collision
        suitable_point_list = []
        for i, (now_point,now_normal) in enumerate(zip(points,normal_vectors)):
          
            result_now_point = result[N_per_point*i:N_per_point*(i+1)]

            angle_list_now_point = angle_list[N_per_point*i:N_per_point*(i+1)]  
            suitable_index = np.where(result_now_point == False)[0]
            if len(suitable_index) > suitable_ratio* N_per_point:
                suitable_point_list.append(now_point)
        
        return np.array(suitable_point_list)



    def generate_grasp_pose(self, points,normal_vectors,collision_points,vis_flag = False):
        #generate the grasp pose
        #points: the object points
        #normal_vectors: the normal vectors of the object points
        #collision_points: the points that the gripper should not collide with
        #return: the grasp pose
        #get the center of the object
        N_per_point = 40
        suitable_ratio = 0.3

        T_list= []
        point_list = []
        angle_list = []
        #generate the possible grasp pose
        for now_point,now_normal in zip(points,normal_vectors):
            #condition about the working space

            angles,rot_mats = generate_T_with_fixed_z(now_normal, N_per_point)

            for angle_,rot_ in zip(angles,rot_mats):
                T_world_2_gripper = np.eye(4)
                T_world_2_gripper[0:3,0:3] = rot_
                T_world_2_gripper[0:3,3] = now_point
                T_list.append(torch.from_numpy(T_world_2_gripper).type(self.d_type))
                point_list.append(now_point)
                angle_list.append(angle_)
        
        collision_points = torch.from_numpy(collision_points).to(self.device) #all points for the object
        collision_points = collision_points.type(self.d_type)

        result = self.check_collision_batch(T_list, collision_points)
        result = result.cpu().numpy()
        collision_points = collision_points.cpu()

        print(result.shape)


        #check the collision
        grasp_pose_T = []
        for i, (now_point,now_normal) in enumerate(zip(points,normal_vectors)):
          
            result_now_point = result[N_per_point*i:N_per_point*(i+1)]

            angle_list_now_point = angle_list[N_per_point*i:N_per_point*(i+1)]  
            suitable_index = np.where(result_now_point == False)[0]
            if len(suitable_index) < suitable_ratio* N_per_point:
                grasp_pose_T.append(None)
            else:
                suitable_angle = [angle_list_now_point[i] for i in suitable_index]
                rotations = R.from_euler('z', suitable_angle)

                average_angle = rotations.mean().as_euler('XYZ')[2]

                #create T
                T_world_2_gripper = np.eye(4)

                original_mat = R.from_euler('z',average_angle).as_matrix()

                dir = np.cross(np.array([0,0,1]), now_normal)
                if np.linalg.norm(dir) < 1e-6:
                    dir = np.array([1,0,0])
                else:
                    dir = dir / np.linalg.norm(dir)
                angle = np.arccos(np.dot(np.array([0,0,1]), now_normal))
                rot_mat = R.from_rotvec(dir * angle).as_matrix()  
                R_world_2_gripper = rot_mat @ original_mat
                T_world_2_gripper[0:3,0:3] = R_world_2_gripper
                T_world_2_gripper[0:3,3] = now_point
                
                collision_res = self.check_collision(T_world_2_gripper, collision_points)
                
                if collision_res == False:
                    grasp_pose_T.append(T_world_2_gripper)
                else:
                    grasp_pose_T.append(None)

        if vis_flag:
            #visualize
            max_vis = 40
            remain_ratio = max_vis/len(grasp_pose_T)
            axis_list = []
            for now_T in grasp_pose_T:
                if now_T is None:
                    continue
                if np.random.rand() > 1-remain_ratio:
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    #find a possible index in tmp result
                    axis.transform(now_T)
                    axis_list.append(axis)


            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(points)

            o3d.visualization.draw_geometries([obj_pcd, *axis_list,axis])

        torch.cuda.empty_cache()
        return grasp_pose_T

        