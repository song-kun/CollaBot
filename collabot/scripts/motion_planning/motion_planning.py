from scipy.spatial.distance import cdist
from cvxopt import solvers, matrix
import copy
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agv_kinematics import Jaka_FK, agv, agv_arm
from rrt_star import RRTStar

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
from util.funcs import xyzrpy_to_T, T_to_xyzrpy, sample_points,path_inter,filter_pcd,path_inter_total_points
def show_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

class motion_planner_agv():
    def __init__(self,agv_pose_list,grasp_list = None,debug = True):
        self.num = len(agv_pose_list)
        self.agv_list = [agv_arm(debug) for i in range(self.num)]

        self.scene_pcd = None
        self.obj_pcd = None #the object that is manipulating
        self.scene_points = None
        self.obj_points = None
        self.obj_pose = None

        #planning part
        self.now_pose = None #now robot pose
        self.now_plan_robot = 0

        self.pcd_ds_size = 0.1

        self.agv_move_cost = 0.5


        self.d_max = 0.5
        self.d_min = 0.05
        self.zeta = 1

        self.dis_tolerance = 0.02
              

        #pose list
        for i in range(self.num):
            self.agv_list[i].move_agv(agv_pose_list[i])
        
        #grasp_list transform from object center to grasp pose
        self.grasp_list = grasp_list

        self.co_checker = None


        self.m = 9 * self.num #num of freedom
        all_center_,all_radius_ = self.agv_list[0].get_collision()
        
        self.n = len(np.concatenate(all_radius_)) #collision for a single robot

        self.velocity_cons = np.array([0.05,0.05,0.05,  0.1,0.1,0.1,0.1,0.1,0.1])
        self.delta_t = 0.1
        self.dq_cons = self.velocity_cons * self.delta_t
        #create velocity cons
        self.G_v_cons = np.concatenate([np.eye(self.m),-np.eye(self.m)],axis = 0)
        self.h_v_cons = np.concatenate([self.dq_cons,self.dq_cons, self.dq_cons,self.dq_cons],axis = 0)



    def assign_scene_obj(self,scene_pcd,obj_pcd,down_sample_obj = True):
        self.scene_pcd = copy.deepcopy(scene_pcd)
        self.obj_pcd = copy.deepcopy(obj_pcd)
        #down sample this two pcd
        self.scene_pcd = self.scene_pcd.voxel_down_sample(voxel_size=self.pcd_ds_size * 5)
        if down_sample_obj:
            self.obj_pcd = self.obj_pcd.voxel_down_sample(voxel_size=self.pcd_ds_size)


        self.scene_points = copy.deepcopy(np.asarray(self.scene_pcd.points))
        self.obj_points = copy.deepcopy(np.asarray(self.obj_pcd.points))

        self.co_checker = collision_manager(self.scene_pcd,self.obj_pcd,grid_size=[0.5,0.5,0.2],dis_th=0.1)

    def get_collsison_free_constraint(self,obj_pose,last_obj_pose = None):
        #static obj pose
        #if last obj pose is None, means the obj is static
        if last_obj_pose is None:
            last_obj_pose = obj_pose

        m = self.m
        n = self.n
            
        G_mat = np.zeros([self.num*n,m])
        h_mat = np.zeros(self.num*n)

        #collect all points for collision check
        all_points = np.array([]).reshape(-1,3)
        all_r = np.array([]).reshape(-1)
        #all jaco list type
        all_jaco = []

        for j in range(self.num):
            #perform motion planning with collision avoidance
            all_c_points,all_c_radius,all_c_jaco = self.agv_list[j].get_colli_points_r_jaco(self.agv_list[j].q)

            all_points = np.concatenate([all_points,np.concatenate(all_c_points)],axis = 0)
            all_r = np.concatenate([all_r,np.concatenate(all_c_radius)],axis = 0)
            all_jaco += all_c_jaco
                    #all points for collision detection
        #transform obj points
        world_obj_points = (obj_pose[0:3,0:3] @ copy.deepcopy(self.obj_points).T).T + obj_pose[0:3,3]
        last_world_obj_points = (last_obj_pose[0:3,0:3] @ copy.deepcopy(self.obj_points).T).T + last_obj_pose[0:3,3]

        # self.scene_points = np.array([0,0,2]).reshape([-1,3]) #TODO

        potential_coll_points = np.concatenate([all_points,world_obj_points, self.scene_points],axis = 0)
        
        
        #0 for collision point, 1 for object, 2 for scene
        robot_point_type = [i for i in range(self.num) for j in range(n)]

        point_type = np.concatenate([np.array(robot_point_type),-np.ones(len(world_obj_points)),-2*np.ones(len(self.scene_points))],axis = 0)
        point_type = point_type.astype(int)
        point_r = np.concatenate([all_r,np.zeros(len(world_obj_points)),np.zeros(len(self.scene_points))],axis = 0)
        #calculate the distance mat
        dis_mat = cdist(all_points,potential_coll_points)

        for i in range(self.num):
            dis_mat[i*self.n:(i+1)*self.n,i*self.n:(i+1)*self.n] = np.inf
        # dis_mat[0:len(all_points),0:len(all_points)] = np.inf #do not consider the collision between the robot
        #find minimal
        residual_dis_mat = dis_mat - all_r.reshape(-1,1) - point_r.reshape(-1)


        # min_dis = np.min(dis_mat,axis = 1)
        # min_index = np.argmin(dis_mat,axis = 1)
        # min_index_type = point_type[min_index]

        min_dis = np.min(residual_dis_mat,axis = 1)
        min_index = np.argmin(residual_dis_mat,axis = 1)
        min_index_type = point_type[min_index]
        # print(np.min(residual_dis_mat[self.n-1]))
        # print(np.argmin(residual_dis_mat[self.n-1]))
        # print(min_index_type[0])

        if np.any(residual_dis_mat < 0.01):
            show_pc(potential_coll_points)
            print("collision happen")


        #update the G_mat and h_mat
        for index, (point_,min_dis_,min_index_,min_index_type_) in enumerate(zip(all_points,min_dis,min_index,min_index_type)):
            #point_: location of the ball
            #min_dis_: minimal distance from this ball to the collision point
            #min_index_: collide with which point
            #min_index_type_: collide with other robot/scene/object
            #r_: r of the collision point

            d = min_dis_ - point_r[min_index_]

            now_point_robot = point_type[index]

            p_r = point_
            p_o = potential_coll_points[min_index_]
            n_ro = p_r - p_o
            n_ro = n_ro / np.linalg.norm(n_ro)
            n_or = -n_ro
            jaco_pr = all_jaco[index]

            if min_index_type_ >= 0:
                #collision with the agv
                jaco_po = all_jaco[min_index_]
                G_mat[index,9*now_point_robot:9*(now_point_robot+1)] = n_ro @ jaco_pr
                G_mat[index,9*min_index_type_:9*(min_index_type_+1)] = n_or @ jaco_po
                h_mat[index] = self.zeta * (d - self.d_min)/(self.d_max - self.d_min)
            elif min_index_type_ == -1:
                #collision with moving object
                delta_p_o = world_obj_points[min_index_ - self.num*n] - last_world_obj_points[min_index_ - self.num*n]
                G_mat[index,9*now_point_robot:9*(now_point_robot+1)] = n_ro @ jaco_pr
                h_mat[index] = self.zeta * (d - self.d_min)/(self.d_max - self.d_min) - n_ro @ delta_p_o

            else:
                #collision with the scene or static object
                G_mat[index,9*now_point_robot:9*(now_point_robot+1)] = n_ro @ jaco_pr
                h_mat[index] = self.zeta * (d - self.d_min)/(self.d_max - self.d_min)

        return G_mat,h_mat
    

    # def motion_planning_ee(self,robots_target_pose_list,obj_pose,k = 2):
    #     #robots_target_pose_list: N * 9
    #     #obj_pose: 4 * 4
    #     #return: N *robot_num 9
    #     targe_q = np.array(robots_target_pose_list).reshape(-1)
    #     robot_pose_list = []
    #     now_pose = [np.array(agv.q) for agv in self.agv_list]
    #     now_pose = np.array(now_pose).reshape(-1)
    #     epsion = 1e-3

    #     m = self.m
        
    #     n = self.n

    #     count = 0

    #     while np.linalg.norm(targe_q - now_pose) > epsion:
    #         # print(count)
    #         # count += 1
    #         # if count % 100 == 0:
    #         #     print(f"norm is {np.linalg.norm(targe_q - now_pose)}")
    #         #     if count > 1000:
    #         #         break
    #         P_mat = np.eye(m)
    #         P_1_mat = np.eye(m)
    #         P_2_mat = np.eye(m)
    #         for j in range(self.num):
    #             P_1_mat[9*j:9*(j+2),9*j:9*(j+2)] *= self.agv_move_cost
            
    #         #min k dqT P dq +  (q+dq-q_target)T P (q+dq-q_target)
    #         P_mat = k*0.5*P_1_mat + P_2_mat

    #         q_mat = 2*(now_pose - targe_q) @ P_2_mat

    #         #collect all points for collision check
    #         G_mat,h_mat = self.get_collsison_free_constraint(obj_pose)

    #         #add velocity cons
    #         G_mat = np.concatenate([G_mat,self.G_v_cons],axis = 0)
    #         h_mat = np.concatenate([h_mat,self.h_v_cons],axis = 0)

    #         solvers.options['show_progress'] = False
    #         # h_mat = 100* h_mat
            
    #         sol = solvers.qp(matrix(P_mat), matrix(q_mat), matrix(G_mat), matrix(h_mat))
    #         delta_q = np.array(sol['x']).reshape(-1)


    #         if np.linalg.norm(delta_q) > 0.1 or np.max(G_mat @ delta_q - h_mat) > 0.001:
    #             print(f"failed in {len(robot_pose_list)}")
    #             print(f"delta q: norm is {np.linalg.norm(delta_q)}, Gx-h {np.max(G_mat @ delta_q - h_mat)}")
    #             return False, robot_pose_list

    #         step_pose= []
    #         for j in range(self.num):
    #             self.agv_list[j].q += delta_q[9*j:9*(j+1)]
    #             step_pose.append(copy.deepcopy(self.agv_list[j].q))
            
    #         now_pose += delta_q

    #         robot_pose_list.append(step_pose)
                
        
    #     #add target
    #     step_pose= []
    #     for j in range(self.num):
    #         self.agv_list[j].move_agv(targe_q[9*j:9*(j+1)]) #modified
    #         step_pose.append(copy.deepcopy(self.agv_list[j].q))
    #     robot_pose_list.append(step_pose)

    #     return True, robot_pose_list

    def check_collision(self,state):
        #state: 9, only one moving robot
        #collect all points for collision check
        robot_pose = state.squeeze(0).numpy()


        self.agv_list[self.now_plan_robot].move_agv(robot_pose)
        this_robot_points, this_robot_r = self.agv_list[self.now_plan_robot].get_collision()
        this_robot_points,this_robot_r = np.concatenate(this_robot_points,axis=0),np.concatenate(this_robot_r,axis=0)

        n = self.n
        other_robot_points = np.array([]).reshape(-1,3)
        other_robot__r = np.array([]).reshape(-1)
        for j in range(self.num):
            if j == self.now_plan_robot: #only move this robot
                continue
            else:
                now_robot_q = self.now_pose[9*j:9*(j+1)]
            self.agv_list[j].move_agv(now_robot_q)
            #get the collision points
            all_c_points, all_c_radius = self.agv_list[j].get_collision()

            other_robot_points = np.concatenate([other_robot_points,np.concatenate(all_c_points)],axis = 0)
            other_robot__r = np.concatenate([other_robot__r,np.concatenate(all_c_radius)],axis = 0)

        #transform obj points
        obj_pose = self.obj_pose
        world_obj_points = (obj_pose[0:3,0:3] @ copy.deepcopy(self.obj_points).T).T + obj_pose[0:3,3]

        potential_coll_points = np.concatenate([other_robot_points,world_obj_points, self.scene_points],axis = 0)

        collision_point_r = np.concatenate([other_robot__r,np.zeros(len(world_obj_points)),np.zeros(len(self.scene_points))],axis = 0)
        #calculate the distance mat

        dis_mat = cdist(this_robot_points,potential_coll_points)

        #find minimal
        residual_dis_mat = dis_mat - this_robot_r.reshape(-1,1) - collision_point_r.reshape(-1)
        
        dis_tolerance = self.dis_tolerance

        if np.min(residual_dis_mat) < dis_tolerance:

            return False
        else:

            return True

    
    def motion_planning_RRT(self,robots_target_pose_list,obj_pose,repeat_time = 4,total_points_num = 1000):
        
        self.obj_pose = obj_pose
        now_pose = np.array([np.array(agv.q) for agv in self.agv_list]).reshape(-1)
        start_pose = np.array([np.array(agv.q) for agv in self.agv_list]).reshape(-1)
        target_pose = np.array(robots_target_pose_list).reshape(-1)

        upper_bound = np.array(self.agv_list[0].upper_bound)
        lower_bound = np.array(self.agv_list[0].lower_bound)

        scene_xyz_min = np.min(self.scene_points,axis = 0)
        scene_xyz_max = np.max(self.scene_points,axis = 0)

        upper_bound[0:2] = scene_xyz_max[0:2]
        lower_bound[0:2] = scene_xyz_min[0:2]
        
        upper_bound = torch.from_numpy(upper_bound).reshape(-1)
        lower_bound = torch.from_numpy(lower_bound).reshape(-1)

        #planning for each robot
        self.now_pose = now_pose
        robot_path_dict = {}
        for i in range(self.num):
            self.now_plan_robot = i
            start = torch.from_numpy(now_pose[9*i:9*(i+1)]).unsqueeze(0)
            goal = torch.from_numpy(target_pose[9*i:9*(i+1)]).unsqueeze(0)

            for _ in range(repeat_time):
                rrt_star = RRTStar(
                    start=start,
                    goal=goal,
                    dim=9,
                    max_state=upper_bound,
                    min_state=lower_bound,
                    expand_dis=1,
                    max_iter=1000,
                    collision_check=self.check_collision,
                    connect_circle_dist=0.2,
                    #2,3,4,5,6,7,8,11,12,13,14,15,16,17
                    rotation_index = [],
                )
                path = rrt_star.planning(debug=False)
                if path is not None:

                    path = path.flipud()
                    path = path.numpy()
                    break
            
            if path is None:
                print(f"Cannot find path for robot {i}")
                return False, []
            else:
                print(f"find path for robot {i}")
                self.now_pose[9*i:9*(i+1)] = target_pose[9*i:9*(i+1)]
                self.agv_list[i].move_agv(target_pose[9*i:9*(i+1)])
                robot_path_dict[i] = path

        new_path_dict = {}
        for i in range(self.num):
            path = robot_path_dict[i]
            new_path = path_inter_total_points(path, total_pts=total_points_num)
            new_path_dict[i] = new_path

        modified_path = []

        current_state = [start_pose[9*i:9*(i+1)] for i in range(self.num)]

        for moving_id in range(self.num):
            traj = new_path_dict[moving_id]

            for step_pose in traj:
                snapshot = copy.deepcopy(current_state)   
                snapshot[moving_id] = step_pose
                modified_path.append(snapshot)

            current_state[moving_id] = traj[-1]

        return True,modified_path

    def motion_planning_ee_follow_path(self,ee_path,obj_pose):  
        #ee_path: N * robot_num * 4 * 4
        #obj_pose:T
        #return: N * robot_num * 9

        robot_pose_list = []

        #using QP to solve the planning problem
        #problem shape
        m = self.m
        n = self.n

        for i in range(1,len(ee_path)):

            # delta_R_list = []
            rot_vec_list = []
            delta_t_list = []
            jaco_list = []

            for j in range(self.num):
                last_pose = ee_path[i-1][j]
                now_pose = ee_path[i][j]
                delta_R = now_pose[:3,:3] @ last_pose[:3,:3].T
                delta_t = now_pose[:3,3] - last_pose[:3,3]

                now_jaco = self.agv_list[j].jacobian(self.agv_list[j].q)

                delta_t_list.append(delta_t)
                rot_vec_list.append(R.from_matrix(delta_R).as_rotvec())
                jaco_list.append(now_jaco)
            
            step_pose =[]
            #planning for each robot
            P_mat = np.eye(m)
            for j in range(self.num):
                P_mat[9*j:9*j+2,9*j:9*j+2] *= self.agv_move_cost

            q_mat = np.zeros(m)
            G_mat = np.zeros([2*n,m])
            h_mat = np.zeros(2*n)
            A_mat = np.zeros([6*self.num,m]) #end pose constrain
            b_mat = np.zeros(6*self.num)
           
            for j in range(self.num):
                #calculate the delta q
                delta_end = np.concatenate([delta_t_list[j],rot_vec_list[j]])
                b_mat[6*j:6*(j+1)] = delta_end
                A_mat[6*j:6*(j+1),9*j:9*(j+1)] = jaco_list[j][:6,:]

            G_mat,h_mat = self.get_collsison_free_constraint(obj_pose)
            solvers.options['show_progress'] = False
            
            sol = solvers.qp(matrix(P_mat), matrix(q_mat), matrix(G_mat), matrix(h_mat), matrix(A_mat), matrix(b_mat))
            delta_q = np.array(sol['x']).reshape(-1)
            
            if np.linalg.norm(delta_q) > 0.1:
                #collision
                print(f"collision in {i}")
                print(f"delta q: norm is {np.linalg.norm(delta_q)}, Gx-h {np.max(G_mat @ np.array(sol['x']) - h_mat)}, Ax-b {np.linalg.norm(A_mat @ np.array(sol['x']) - b_mat)}")

                return False, robot_pose_list

            for j in range(self.num):
                self.agv_list[j].q += delta_q[9*j:9*(j+1)]
                step_pose.append(copy.deepcopy(self.agv_list[j].q))
            
            robot_pose_list.append(step_pose)
        
        return True, robot_pose_list


    def add_grasp(self,T_world_2_obj):
        #using the current state as grasp pose
        #add grasp
        grasp_list = []
        for i in range(self.num):
            robot_end = self.agv_list[i].get_ee_T()
            grasp_list.append(np.linalg.inv(T_world_2_obj) @ robot_end)
        self.grasp_list = grasp_list
        # robot1_end = planner.agv_list[0].get_ee_T()
        # robot2_end = planner.agv_list[1].get_ee_T()

        # planner.grasp_list = [np.linalg.inv(T_world_2_obj) @ robot1_end, np.linalg.inv(T_world_2_obj) @ robot2_end]
        
    def motion_planning(self,object_pose_list):
        #motion planning with end constrain
        #object_pose_list: (N,)
        #remember set the initial state to be feasible
        robot_pose_list = []

        #using QP to solve the planning problem
        #problem shape
        m = self.m #total dof
        n = self.n #collision ball for one robot

        step_pose= []
        for j in range(self.num):
            step_pose.append(copy.deepcopy(self.agv_list[j].q))
        robot_pose_list.append(step_pose)

        for i in range(1,len(object_pose_list)):
            
            last_pose = object_pose_list[i-1]
            now_pose = object_pose_list[i]
            #get the grasp point pose
            last_grasp_pose = [last_pose @ grasp for grasp in self.grasp_list]
            now_grasp_pose = [now_pose @ grasp for grasp in self.grasp_list]
            #calculate the delta R and delta t for each robot

            rot_vec_list = []
            delta_t_list = []
            jaco_list = []
            for j in range(self.num):
                last_R = last_grasp_pose[j][:3,:3]
                last_t = last_grasp_pose[j][:3,3]
                now_R = now_grasp_pose[j][:3,:3]
                now_t = now_grasp_pose[j][:3,3]
                delta_R = now_R @ last_R.T
                delta_t = now_t - last_t
                rot_vec_list.append(R.from_matrix(delta_R).as_rotvec())
                delta_t_list.append(delta_t)

                #calculate the jacobian matrix
                jaco_list.append(self.agv_list[j].jacobian(self.agv_list[j].q))
            
            step_pose =[]
            #planning for each robot
            P_mat = np.eye(m)
            for j in range(self.num):
                P_mat[9*j:9*j+3,9*j:9*j+3] *= self.agv_move_cost
            q_mat = np.zeros(m)
            # G_mat = np.zeros([2*n,m])
            # h_mat = np.zeros(2*n)
            A_mat = np.zeros([6*self.num,m]) #end pose constrain
            b_mat = np.zeros(6*self.num)

            #end constrain         
            for j in range(self.num):
                #calculate the delta q
                delta_end = np.concatenate([delta_t_list[j],rot_vec_list[j]])
                b_mat[6*j:6*(j+1)] = delta_end
                A_mat[6*j:6*(j+1),9*j:9*(j+1)] = jaco_list[j][:6,:]            

            #collision free constrain
            G_mat,h_mat = self.get_collsison_free_constraint(now_pose,last_pose)


            solvers.options['show_progress'] = False
            
            sol = solvers.qp(matrix(P_mat), matrix(q_mat), matrix(G_mat), matrix(h_mat), matrix(A_mat), matrix(b_mat))
            
            delta_q = np.array(sol['x']).reshape(-1)

            if np.linalg.norm(delta_q) > 0.1:
                #collision
                print(f"collision in step {i}")
                print(f"delta q: norm is {np.linalg.norm(delta_q)}, Gx-h {np.max(G_mat @ np.array(sol['x']) - h_mat)}, Ax-b {np.linalg.norm(A_mat @ np.array(sol['x']) - b_mat)}")

                return False, robot_pose_list

            for j in range(self.num):
                self.agv_list[j].q += delta_q[9*j:9*(j+1)]
                step_pose.append(copy.deepcopy(self.agv_list[j].q))
            
            robot_pose_list.append(step_pose)
                #add target

        return True, robot_pose_list
    
    def animation(self,qs,ts = None,dt = 0.05,scene_pcd=None,object_list = None,obj_pose_list = None, stop_time = 1,image_dir = None,
                  look_at = [0,-2,0],zoom=0.35):
        #qs:M * N * 9
        #ts: M *1
        #M: frame; N: number of robot;  9: 3 for agv, 6 for jaka
        if ts is None:
            ts = [1 for i in range(len(qs))]

        new_obj_list = copy.deepcopy(object_list)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        
        if scene_pcd is not None:
            vis.add_geometry(scene_pcd)
        origin_obj_points_list = []
        if object_list is not None:

            for obj in new_obj_list:
                origin_obj_points_list.append(copy.deepcopy(np.asarray(obj.points)))
                vis.add_geometry(obj)
                
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
        # vis.add_geometry(coordinate_frame)



        agv_pcd_list = [o3d.geometry.PointCloud() for i in range(self.num)]
        for i in range(self.num):
            agv_points,arm_points = self.agv_list[i].vis_collision(show=False)
            #concate
            total_points = np.concatenate([agv_points,arm_points],axis = 0)
            agv_pcd_list[i].points = o3d.utility.Vector3dVector(total_points)
            vis.add_geometry(agv_pcd_list[i])
        
        view_control = vis.get_view_control()

        view_control.set_front([0,0,1])  
        view_control.set_up([ 1,0,0 ])   
        view_control.set_lookat(look_at)            
        view_control.set_zoom(zoom)  

        #update
        index = 0 
        for q,t in zip(qs,ts):
            for i in range(self.num):
                self.agv_list[i].move_agv(q[i])
                agv_points,arm_points = self.agv_list[i].vis_collision(show=False)
                new_points = copy.deepcopy(np.concatenate([agv_points,arm_points],axis = 0))
                # agv_pcd_list[i].points = o3d.utility.Vector3dVector(new_points)
                np.asarray(agv_pcd_list[i].points)[:] = new_points
                vis.update_geometry(agv_pcd_list[i])
            
            if obj_pose_list is not None:
                for i, obj in enumerate(new_obj_list):
                    obj_pose = obj_pose_list[index]
                    origin_point = origin_obj_points_list[i]
                    # obj.points = o3d.utility.Vector3dVector(np.matmul(obj_pose[:3,:3], origin_point.T).T + obj_pose[:3,3])
                    np.asarray(obj.points)[:] = copy.deepcopy(np.matmul(obj_pose[:3,:3], origin_point.T).T + obj_pose[:3,3])

                    vis.update_geometry(obj)
            #update object
            vis.poll_events()
            vis.update_renderer()
            if image_dir is not None:
                # Save the current view to a file in each iteration
                image_path = os.path.join(image_dir, f"capture_{index:04d}.png")
                vis.capture_screen_image(image_path, do_render=False)

            index += 1
            time.sleep(dt)
        for i in range(100):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(stop_time/100)
        vis.destroy_window()
    
    #create sdf for the scene
    def obj_planning(self,start_np,goal_np,n_point = 200):
        #start_np: 6
        #goal_np: 6
        #return path: N * 6
        start = torch.from_numpy(start_np).unsqueeze(0)
        goal = torch.from_numpy(goal_np).unsqueeze(0)
        

        max_offset = 100
        max_z = 1.8
        max_state = torch.tensor([max_offset, max_offset,max_z,torch.pi,torch.pi,torch.pi])
        min_state = torch.tensor([-max_offset, -max_offset,-max_z,-torch.pi,-torch.pi,-torch.pi])


        rrt_star = RRTStar(
            start=start,
            goal=goal,
            dim=6,
            max_state=max_state,
            min_state=min_state,
            expand_dis=0.5,
            max_iter=1000,
            collision_check=self.co_checker.check_collision,
            connect_circle_dist=1.0,
            rotation_index = [],
        )
        path = rrt_star.planning(debug=False)
        if path is None:
            print("Cannot find path")
            return False,[]
        else:
            path = path.flipud()
        path = path.numpy()



        path = path_inter_total_points(path,n_point)

        return True,path
    

    def obj_planning_keepz(self,start_np,goal_np,n_point = 200):
        #start_np: 4
        #goal_np: 4
        #return path: N * 4
        start = torch.from_numpy(start_np).unsqueeze(0)
        goal = torch.from_numpy(goal_np).unsqueeze(0)
        

        max_offset = 100
        max_z = 1.8
        max_state = torch.tensor([max_offset, max_offset,max_z,torch.pi])
        min_state = torch.tensor([-max_offset, -max_offset,-max_z,-torch.pi])


        rrt_star = RRTStar(
            start=start,
            goal=goal,
            dim=4,
            max_state=max_state,
            min_state=min_state,
            expand_dis=0.5,
            max_iter=1000,
            collision_check=self.co_checker.check_collision,
            connect_circle_dist=1.0,
            rotation_index = [],
        )
        path = rrt_star.planning(debug=False)
        if path is None:
            print("Cannot find path")
            return False,[]
        else:
            path = path.flipud()
        path = path.numpy()



        path = path_inter_total_points(path,n_point)

        return True,path



class collision_manager:
    def __init__(self,scene,obj,dis_th = 0.1, d_type = torch.float32,grid_size=[1,1,1]):
        #check collision between the obj and the scene
        self.dis_th = dis_th
        self.d_type = d_type
        self.scene = scene
        self.scene_points = np.asarray(scene.points)
        self.scene_points = torch.from_numpy(self.scene_points).type(self.d_type)
        self.obj = obj
        self.obj_points = np.asarray(obj.points)
        self.obj_points = torch.from_numpy(self.obj_points).type(self.d_type)

        #delete the object points with small z value
        # min_obj_z = torch.min(self.obj_points[:,2])
        min_z = torch.min(self.scene_points[:,2])
        self.obj_points = self.obj_points[self.obj_points[:,2]>min_z+self.dis_th + 0.01]


        self.min_xyz = [torch.min(self.scene_points[:,i]) for i in range(3)]
        self.max_xyz = [torch.max(self.scene_points[:,i]) for i in range(3)]

        #using grid size of 3 different size
        self.grid_size = grid_size
        self.x = torch.arange(self.min_xyz[0], self.max_xyz[0], self.grid_size[0])
        self.y = torch.arange(self.min_xyz[1], self.max_xyz[1], self.grid_size[1])
        self.z = torch.arange(self.min_xyz[2], self.max_xyz[2], self.grid_size[2])
        self.mesh_x, self.mesh_y, self.mesh_z = torch.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.grid_point = torch.stack([self.mesh_x, self.mesh_y, self.mesh_z], dim=-1).reshape(-1, 3).type(self.d_type) # (m,3)
        self.sdf = torch.cdist(self.grid_point, self.scene_points).min(dim=1).values

    def min_dis_sdf(self,now):
        now = now.reshape(-1,3)
        dis = torch.cdist(now,self.grid_point)

        min_index = torch.argmin(dis,dim=1)
        min_dis = self.sdf[min_index]

        return min_dis

    def vis_scene_obj(self):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])

        o3d.visualization.draw_geometries([self.scene,self.obj,coordinate_frame])

    def check_collision(self,state):
        #state: x,y,z,r,p,y
        #return true for not collision
        state = state.squeeze(0)
        if len(state) != 6:
            r = R.from_euler('z', state[3]).as_matrix()
        else:
            r = R.from_euler('xyz', state[3:]).as_matrix()
        r = torch.from_numpy(r).type(self.d_type)
        obj_points = torch.matmul(r,self.obj_points.T).T+ state[:3]


        min_dis = self.min_dis_sdf(obj_points)

        
        if torch.min(min_dis) < self.dis_th: #collision with the env
            return False
        else:
            return True
    
        


def load_test_pcd():
    #load the pcd file
    #world frame x y is the same as the gazebo, z is the same height as the robot base frame

    current_path = os.path.dirname(os.path.abspath(__file__))

    scene_path = os.path.join(current_path, '../../example/pc_data/scene.ply')
    table_path = os.path.join(current_path, '../../example/pc_data/table.ply')
    #load two model
    scene = o3d.io.read_point_cloud(scene_path)
    table = o3d.io.read_point_cloud(table_path)

    scene = filter_pcd(scene)
    table = filter_pcd(table)

    #get the object center
    object_center = table.get_center()
    T_world_2_obj = np.eye(4)
    T_world_2_obj[0:2,3] = object_center[0:2]

    min_z = np.min(np.array(scene.points)[:,2])

    #offset scene and table
    #in simulation, the z axis pose of robot is 0.2
    scene_points = np.array(scene.points)
    scene_points[:,2] = scene_points[:,2] - min_z - 0.2 #to avoid collision with the robot
    scene.points = o3d.utility.Vector3dVector(scene_points)

    table_points = np.array(copy.deepcopy(table.points))
    table_points[:,2] = table_points[:,2] - min_z - 0.2
    table.points = o3d.utility.Vector3dVector(table_points)

    origin_trans = np.eye(4)
    origin_trans[0,3] = object_center[0]
    origin_trans[1,3] = object_center[1]

    #move the object to the center
    new_table_points = copy.deepcopy(table_points)
    new_table_points[:,0] -= object_center[0]
    new_table_points[:,1] -= object_center[1]
    new_table_pcd = o3d.geometry.PointCloud()
    new_table_pcd.points = o3d.utility.Vector3dVector(new_table_points) #new object
    new_table_pcd.colors = table.colors

    return scene,new_table_pcd,T_world_2_obj

def interpolation_T(T1,T2,n_step):
    #T1: 4 * 4
    #T2: 4 * 4
    #return: n_step * 4 * 4
    R1 = T1[:3,:3]
    R2 = T2[:3,:3]
    t1 = T1[:3,3]
    t2 = T2[:3,3]

    Rot_vec1 = R.from_matrix(R1).as_rotvec()
    delta_R = R.from_matrix(R2 @ R1.T).as_rotvec()
    delta_t = t2 - t1

    T_list = []
    for i in range(n_step):
        ratio = i/n_step
        now_R = R.from_rotvec(Rot_vec1 + delta_R * ratio).as_matrix()
        now_t = t1 + delta_t * ratio
        now_T = np.eye(4)
        now_T[:3,:3] = now_R
        now_T[:3,3] = now_t
        T_list.append(now_T)
    
    return T_list

def merge_pcd(pcd_list):
    #pcd_list: N * pcd
    #return: merged pcd
    new_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        new_pcd += pcd
    return new_pcd

def create_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

if __name__ == "__main__":
    '''
    example for motion planning
    1. define the grasp point for each robot, in this demo, we use the left and right point of the table
    2. move the robot to the grasp point
    3. define the object start and target pose, in the real implement, this should be generated by LLM
    4. planning for each robot while keeping the end effector constraint
    '''
    scene,table,T_world_2_obj = load_test_pcd()

    #current pose of two agv
    agv_pose_list = [[-1,-6,0,0, 1, 0, -1, -1.57, 0], [3.5,-6,3.14,0, 1, 0, -1, -1.57, 0]]
    agv_pose_list = [[1.267155884915246, 0.46931527532146444, -0.005824006290116761, 0, 0, 0, 0, 0, 0], [0.42473332673917386, -2.0055206619231796, -0.427197799624819, 0, 0, 0, 0, 0, 0]]

    # agv_pose_list = [[1.267155884915246, 0.46931527532146444, -0.005824006290116761, 0, 0, 0, 0, 0, 0], 
    #                  [0.42473332673917386, -2.0055206619231796, -0.427197799624819, 0, 0, 0, 0, 0, 0],
    #                  [0,0,-0.427197799624819, 0, 0, 0, 0, 0, 0]]

   
    planner = motion_planner_agv(agv_pose_list)
    planner.assign_scene_obj(scene,table)
    #update the agv pose
    for i in range(len(agv_pose_list)):
        planner.agv_list[i].move_agv(agv_pose_list[i])

    #SETP 1: generate a feasible pose for the grasp pose
    grasp_pose1 = [-7.29017956e-01, -6.18884320e+00, -4.07518369e-02, -1.62116320e-02,  4.57385640e-01,  3.12432298e-02, -4.88619949e-01, -1.62696284e+00,  7.54421594e-05]
    grasp_pose2 = [ 2.62535179e+00, -6.68710033e+00,  3.25923487e+00,  3.10566995e-02,  4.77450818e-01 , 2.89129749e-02, -5.06365248e-01 ,-1.41970842e+00,  4.96827067e-06]

    grasp_pose1 = [ 1.1445704,  -6.94871834 ,-2.44346095 ,-0.57963039 , 3.09158487 ,-1.37644894 , -0.48444757 , 0.3198795 ,  0.4842905 ]
    grasp_pose2 = [ 1.00984529, -5.93684098 , 0.34906585 ,-0.40020172 , 3.01898005, -2.54814576, 3.91169206, -0.12328851, -3.08185062]
    grasp_pose = np.array(grasp_pose1 + grasp_pose2)

    grasp_pose1 = [-7.29017956e-01, -6.18884320e+00, -4.07518369e-02, -1.62116320e-02,  4.57385640e-01,  3.12432298e-02, -4.88619949e-01, -1.62696284e+00,  7.54421594e-05]
    grasp_pose2 = [ 2.62535179e+00, -6.68710033e+00,  3.25923487e+00,  3.10566995e-02,  4.77450818e-01 , 2.89129749e-02, -5.06365248e-01 ,-1.41970842e+00,  4.96827067e-06]
    grasp_pose3 = [ 1.1445704,  -6.94871834 ,-2.44346095 ,-0.57963039 , 3.09158487 ,-1.37644894 , -0.48444757 , 0.3198795 ,  0.4842905 ]
    grasp_pose = np.array(grasp_pose1 + grasp_pose2 )

    # STEP 2: motion planning from robot current pose to grasp pose
    # success, robot_pose1 = planner.motion_planning_RRT(grasp_pose,T_world_2_obj) #motion planning will influence the robot_pose
    # if  success:
    #     print(f"success in grasping process with {len(robot_pose1)} steps")
    #     obj_pose_list = [T_world_2_obj for i in range(len(robot_pose1))]
    #     planner.animation(robot_pose1,dt=0.01,scene_pcd=scene,object_list=[table],obj_pose_list=obj_pose_list, stop_time=1)
    # else:
    #     print("failed in grasping process")
    
    
    planner.agv_list[0].move_agv(grasp_pose1)
    planner.agv_list[1].move_agv(grasp_pose2)
    # planner.agv_list[2].move_agv(grasp_pose3)
    planner.add_grasp(T_world_2_obj)

    #STEP 3: planning for the object
    start = np.array([T_world_2_obj[0,3],T_world_2_obj[1,3],0,0,0,0]) #x,y,z,r,p,y
    goal = np.array([0,0,0.2,0,0,1.57]) #move to origin
    success, path = planner.obj_planning(start,goal)
    obj_path_T = [xyzrpy_to_T(p,xyz_first=True) for p in path] #in world frame
    if success:
        print(f"find object path with {len(path)} points")

    #STEP 4: motion planning for each robot
    success, robot_pose2 = planner.motion_planning(obj_path_T)
    if  success:
        print("success in planning for each robot")
        planner.animation(robot_pose2,scene_pcd=scene,object_list=[table],obj_pose_list=obj_path_T,stop_time=2)
    else:
        print("failed in planning for each robot")
        planner.animation(robot_pose2,scene_pcd=scene,object_list=[table],obj_pose_list=obj_path_T,stop_time=10)
