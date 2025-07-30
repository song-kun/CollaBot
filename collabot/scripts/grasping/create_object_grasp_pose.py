import numpy as np
import torch
import sys
import os
import copy
import open3d as o3d
import numpy as np
import json
import re
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import itertools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pointnet_cls import LoGNet_pn
from grasp_functions import RigidTransform,vis_grasp_pose,vis_grasp_poses,load_ply_mesh,viewpoint_params_to_matrix,turn_o3d_into_input
from check_grasp_collision import grasper_model

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))   ,'../'))

from util.pc_data_utils import down_sample,show_pc,simple_remove_ground,remove_outlier,extract_cluster_centers
from PIL import Image
import cv2
from util.prompt_interface import prompt_interface
from scipy.spatial import cKDTree
from agv_ik import AGVIK
import time
_PROXY_VARS = [
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "ALL_PROXY", "all_proxy", "no_proxy", "NO_PROXY"
]

for v in _PROXY_VARS:
    os.environ.pop(v, None)     
from scipy.spatial import distance

class generate_grasp_pose:
    def __init__(self,device="cuda",input_pc_dim=3,model_path = "./trained_model/last_390.pth",gripper_mesh_path=None):
        self.device = device
        self.input_pc_dim = input_pc_dim
        self.test_model = LoGNet_pn(n_bin=1)
        self.test_model.eval()
        self.input_point_num = 2048
        
        state_dict = torch.load(model_path,weights_only=True)
        self.test_model.load_state_dict(state_dict)
        if gripper_mesh_path is None:
            print("failed in load gripper mesh")
            self.gripper_mesh = None
        else:
            self.gripper_mesh = self.load_gripper_mesh(gripper_mesh_path)
        

    def load_gripper_mesh(self,mesh_path):
        gripper_mesh = load_ply_mesh(mesh_path,debug=False)
        return gripper_mesh
    
    def generate_grasp_pose(self,pc,points,vis_flag = True):
        #pc: n*6 array, first 3 point location, last 3 normal vectors
        #points: n*3 array, point location to generate grasp pose

        if len(pc) < self.input_point_num:
            print("not enough point")
            return []
        
        #move pc to center
        center = np.mean(pc, axis=0)
        centered_pc  = pc - center

        now_obj = o3d.geometry.PointCloud()
        now_obj.points = o3d.utility.Vector3dVector(centered_pc)
        total_pc = []
        for now_point in points:
            input_pc = turn_o3d_into_input(now_obj,now_point - center,box_size=1,vis_flag=False)
            total_pc.append(input_pc)

        total_pc = np.array(total_pc)

        with torch.no_grad():
            
            total_pc_torch = torch.from_numpy(copy.deepcopy(total_pc[:,0:self.input_pc_dim,:])).float().to(self.device)
            
            self.test_model=self.test_model.to(self.device)
            approach_,angle_,trans_ = self.test_model(total_pc_torch)

            approach_ = approach_.detach().cpu().numpy()
            angle_ = angle_.detach().cpu().numpy()
            trans_ = trans_.detach().cpu().numpy()

        generated_grasp_pose = []

        for i in range(len(points)):
            now_approach = approach_[i]

            now_angle = angle_[i][0]

            now_trans = trans_[i]
            now_R = viewpoint_params_to_matrix(now_approach,now_angle)
            grasp_pose = RigidTransform(now_R,now_trans)
            if vis_flag:
                vis_grasp_pose(copy.deepcopy(total_pc[i]),grasp_pose,self.gripper_mesh)
            
            generated_grasp_pose.append(grasp_pose)
        
        self.test_model = self.test_model.to("cpu")

        #to origin frame
        world_poses = []
        for grasp_point, grasp_pose in zip(points,generated_grasp_pose):
            world_poses.append(RigidTransform(grasp_pose[0:3,0:3],grasp_pose[0:3,3] + grasp_point))
        
        return world_poses
        # return [RigidTransform(grasp_pose[0:3,0:3],grasp_pose[0:3,3] + center) for grasp_pose in generated_grasp_pose]


class large_obj_grasping:
    def __init__(self,img_list,pc_list,camera_pose_list,gripper_CAD_path, target_grasp_pose_num = 10,n_robot = 2,
                 api_type="qwen",VLM_model = "qwen-vl-plus-2025-01-25"):
        """
        input: point clouds and camera poses
        goal: generate N grasp pose

        Note: point cloud in the world frame
        """
        self.img_list = img_list
        self.pc_list = pc_list
        self.camera_pose_list = camera_pose_list
        self.target_grasp_pose_num = target_grasp_pose_num
        self.n_robot = n_robot

        #merge the pc
        self.merged_pc = simple_remove_ground(down_sample(np.concatenate(pc_list,axis=0)))
        self.merged_pc = remove_outlier(self.merged_pc)
        

        #local grasp
        ply_file = os.path.join(gripper_CAD_path,"gripper.PLY")  # path for gripper mesh
        model_path = os.path.join(gripper_CAD_path,"trained_model/lognet.pth")
        self.grasp_pose_generator = generate_grasp_pose(gripper_mesh_path=ply_file,model_path=model_path)

        #load intrinsic
        intri_path = os.path.join(gripper_CAD_path,'../../param/intrinsics.npy')
        self.intrinsic = np.load(intri_path)

        #LLM part
        api_file_path = os.path.join(gripper_CAD_path, f'../../prompts/api_{api_type}.json')
        with open(api_file_path, "r") as f:
            config = json.load(f)
        self.VLM_model = VLM_model

        self.api_key = config["api_key"]
        self.api_base = config["api_base"]
        self.prompt_inter = prompt_interface(self.api_key,self.api_base)
        choose_grasp_pose_prompt_path = os.path.join(gripper_CAD_path, '../../prompts/grasp_point_prompts.txt')
        with open(choose_grasp_pose_prompt_path, "r") as f:
            self.choose_grasp_pose_prompt = f.read()
        
        #AGV IK part
        self.ik_solver = AGVIK()
        self.agv_base_height = self.ik_solver.agv_arm.agv_base_height



    def genera_feasible_grasp_poses(self,img_index,collision_points,vis_grasp_flag = True):
        """
        generate grasp poses that can be solved using IK
        """
        now_pc_o3d = o3d.geometry.PointCloud()
        now_pc_o3d.points = o3d.utility.Vector3dVector(self.merged_pc)
        point_cloud = now_pc_o3d

        # estimate the normal vectors
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10)
        )
        point_cloud.orient_normals_consistent_tangent_plane(100)

        now_grasp = grasper_model(5,device='cuda')

        points = copy.deepcopy(now_pc_o3d.points)
        normal_vectors = np.asarray(now_pc_o3d.normals)
        # collision_points = np.asarray(now_pc_o3d.points)
        print(np.min(collision_points,axis=0))
        

        all_grasp_points = now_grasp.generate_potential_grasp_points(points,normal_vectors,collision_points)
        

        #find grasp points in this image
        now_obj_points = copy.deepcopy(self.pc_list[img_index])

        tree = cKDTree(now_obj_points)
        distances, _ = tree.query(all_grasp_points, k=1) 

        from_same_sample_th = 0.01
        in_img_grasp_points = all_grasp_points[distances < from_same_sample_th]
        # show_pc(in_img_grasp_points)

        now_potential_grasp_points = copy.deepcopy(in_img_grasp_points)
        max_iter = 3
        delete_th = 0.05

        for _ in range(max_iter):
            
            #generate M grasp point
            grasp_points = extract_cluster_centers(now_potential_grasp_points,self.target_grasp_pose_num)

            grasp_pose_list = self.grasp_pose_generator.generate_grasp_pose(now_obj_points,grasp_points,vis_flag= False)

            if vis_grasp_flag:
                vis_grasp_poses(copy.deepcopy(now_obj_points.T),grasp_pose_list,gripper_mesh=self.grasp_pose_generator.gripper_mesh)
            
            #IK Check
            feasible_flag = []
            for now_grasp_pose in grasp_pose_list:
                grasp_q = self.agv_ik_solver(now_grasp_pose,collision_points=collision_points,vis_grasp_pose=False)
                if grasp_q is None:
                    feasible_flag.append(False)
                else:
                    feasible_flag.append(True)
            #if all poses are feasible, break
            if np.all(feasible_flag):
                break
            #re-generate grasp poses
            #delete grasp point near these points
            for now_index in range(len(feasible_flag)):
                if feasible_flag[now_index]:
                    continue
                #delete the grasp point
                now_grasp_point_ = grasp_points[now_index]
                #calculate the distance between grasp_point and all potential grasp points
                distances = np.linalg.norm(now_potential_grasp_points - now_grasp_point_, axis=1)
                #delete the points that are too close
                now_potential_grasp_points = now_potential_grasp_points[distances > delete_th]
        
        print(feasible_flag)
        return grasp_pose_list,feasible_flag






        

        
    def generate_grasp_point(self,img_index):
        """
        generate M grasp points from the merged point cloud, M is target_grasp_pose_num
        """
        now_pc_o3d = o3d.geometry.PointCloud()
        now_pc_o3d.points = o3d.utility.Vector3dVector(self.merged_pc)
        point_cloud = now_pc_o3d

        # estimate the normal vectors
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10)
        )
        point_cloud.orient_normals_consistent_tangent_plane(100)

        now_grasp = grasper_model(5,device='cuda')

        points = copy.deepcopy(now_pc_o3d.points)
        normal_vectors = np.asarray(now_pc_o3d.normals)
        collision_points = np.asarray(now_pc_o3d.points)
        

        all_grasp_points = now_grasp.generate_potential_grasp_points(points,normal_vectors,collision_points)
        

        #find grasp points in this image
        now_obj_points = self.pc_list[img_index]
        tree = cKDTree(now_obj_points)
        distances, _ = tree.query(all_grasp_points, k=1) 

        from_same_sample_th = 0.01
        in_img_grasp_points = all_grasp_points[distances < from_same_sample_th]
        # show_pc(in_img_grasp_points)
        
        #generate M grasp point
        grasp_points = extract_cluster_centers(in_img_grasp_points,self.target_grasp_pose_num)
        

        return grasp_points


    def pose_generate(self,img_index,grasp_points,vis_grasp_flag = True):
        now_obj_points = self.pc_list[img_index]

        grasp_pose_list = self.grasp_pose_generator.generate_grasp_pose(now_obj_points,grasp_points,vis_flag= False)

        if vis_grasp_flag:
            vis_grasp_poses(copy.deepcopy(now_obj_points.T),grasp_pose_list,gripper_mesh=self.grasp_pose_generator.gripper_mesh)

        return grasp_pose_list




    def create_project_img(self,texted_img,center_poses,extrinsic_matrix,intrinsic_matrix,output_path=None,mask = None):
        if np.max(texted_img) < 2:
            texted_img = (texted_img * 255).astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1

        bg_padding = 1
        for i, now_center in enumerate(center_poses):
            center_t = now_center[0:3,3]
            uv = intrinsic_matrix @ (extrinsic_matrix[0:3,0:3] @ center_t + extrinsic_matrix[0:3,3])
            uv = uv / uv[2]  

            # text on image
            text = f"{i}"
            coordinates = (int(uv[0]), int(uv[1]))  
            text = f"{i}"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_origin = (int(uv[0]) - text_size[0] // 2, int(uv[1]) + text_size[1] // 2)

            # draw rectangle background
            cv2.rectangle(texted_img, (text_origin[0] - bg_padding, text_origin[1] - text_size[1] - bg_padding), 
                        (text_origin[0] + text_size[0] + bg_padding, text_origin[1] + bg_padding), ( 242,215, 213), cv2.FILLED)
            
            # draw text
            cv2.putText(texted_img, text, text_origin, font, font_scale, (23, 32, 42), font_thickness, cv2.LINE_AA)

        # save
        padding = 20
        if mask is not None:
            #only keep the rectangle that contain the mask 
            left_top = np.min(np.where(mask == 1),axis = 1)
            right_bottom = np.max(np.where(mask == 1),axis = 1)
            left_top[0] = max(left_top[0] - padding,0)
            left_top[1] = max(left_top[1] - padding,0)
            right_bottom[0] = min(right_bottom[0] + padding,mask.shape[0])
            right_bottom[1] = min(right_bottom[1] + padding,mask.shape[1])
            new_img = texted_img[left_top[0]:right_bottom[0],left_top[1]:right_bottom[1]]
            if output_path is not None:
                cv2.imwrite(output_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        else:
            new_img = texted_img
            if output_path is not None:
                cv2.imwrite(output_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
        
        return new_img

    def choose_grasp_pose(self,img_index,gras_points,language_instruction):
        #choose the grasp pose that is not occluded by other objects
        #img_index: the index of the image
        #gras_points: the grasp points in the world frame

        now_img = self.img_list[img_index]
        now_camer_pose = self.camera_pose_list[img_index]
        extrinsic_matrix = np.linalg.inv(now_camer_pose)
        create_img = self.create_project_img(now_img,gras_points,extrinsic_matrix,self.intrinsic,output_path='./test_project.png',mask = None)
        #using LLM to choose grasp pose
        now_prompt = copy.deepcopy(self.choose_grasp_pose_prompt)

        values = {
            "num_robot": self.n_robot,
            "task_description": language_instruction,
            "language_instruction": language_instruction,
            "num_grasp": len(gras_points),
            "num_grasp-1": len(gras_points) - 1
        }

        def replace_var(match):
            var_name = match.group(1)
            return str(values.get(var_name, match.group(0))) 

        modifed_prompt = re.sub(r"\$\((.*?)\)", replace_var, now_prompt)
        #stream: qwen3-235b-a22b
        #not stream: qwen-vl-plus-2025-01-25
        res = self.prompt_inter.base_interface(create_img,modifed_prompt,model=self.VLM_model)
        
        match = re.search(r'MY CHOICE IS:\s*\[(.*?)\]', res)
        if match:
            choice_str = match.group(1)
            choice_list = [int(x.strip()) for x in choice_str.split(',')]
        else:
            choice_list = []
            print("No match found.")
        
        return choice_list
    
    def from_grasp_pose_to_tool0(self,grasp_pose):
        #rotation
        R_grasp_2_tool0 = np.array([[0,0,1],[0,-1,0],[1,0,0]])
        t_grasp_2_tool0 = np.array([-self.ik_solver.agv_arm.wrist3_2_ee[2,3],0,0])
        T_grasp_2_tool0 = RigidTransform(R_grasp_2_tool0,t_grasp_2_tool0)

        tool0_frame = grasp_pose @ T_grasp_2_tool0
        x_axis = tool0_frame[0:3,0]
        if np.dot(x_axis,[0,0,1]) < -0.8:
            #adjust the pose of camera
            # print("reverse grasp pose")
            rot_mat = R.from_euler('z',np.pi, degrees=False).as_matrix()
            tool0_frame[0:3,0:3] =  tool0_frame[0:3,0:3] @ rot_mat
        
        return tool0_frame
    
    def collaborative_ik_solver(self,grasp_poses,collision_points=None,vis_grasp_pose = True,max_potential_q = 100):
        all_grasp_poses = []

        for now_grasp_pose in grasp_poses:
            qs = self.agv_ik_solver(now_grasp_pose,collision_points=collision_points,vis_grasp_pose=False,select_flag=False)

            if len(qs) == 0:
                print("no solution for this grasp pose")
                return None
            if len(qs) > max_potential_q:
                random_seleted = np.random.choice(len(qs), max_potential_q, replace=False)
                all_grasp_poses.append([qs[i] for i in random_seleted])
            else:
                all_grasp_poses.append(qs)
        
        
        all_joint_combinations = list(itertools.product(*all_grasp_poses))

        print(f"Total combinations to evaluate: {len(all_joint_combinations)}")
        print(len(all_joint_combinations))

        # Step 3: Evaluate combinations and select a valid one
        for joint_combination in all_joint_combinations:
            if self.check_collision_between_robot(joint_combination):
                print("Found a valid joint combination for all robots.")
                break
        
        merged_pc_o3d = o3d.geometry.PointCloud()
        # merged_pc_o3d.points = o3d.utility.Vector3dVector(self.merged_pc)
        merged_pc_o3d.points = o3d.utility.Vector3dVector(collision_points)
        show_o3d = copy.deepcopy(merged_pc_o3d)
        show_points = np.array(show_o3d.points)
        show_points[:,2] -= self.agv_base_height
        show_o3d.points = o3d.utility.Vector3dVector(show_points)
        for choose_q in joint_combination:
            T_fk = self.ik_solver.agv_arm.FK(choose_q)
            if vis_grasp_pose:
                self.ik_solver.agv_arm.vis_collision(show_obj_list = [show_o3d],vis_coordinate=False)
        
        return joint_combination


    def check_collision_between_robot(self,qs):
        all_C = np.array([]).reshape(-1,3)
        all_R = np.array([]).reshape(-1,1)
        for now_q in qs:
            #check collision with the scene
            T_fk = self.ik_solver.agv_arm.FK(now_q)
            collision_center,collision_r =  self.ik_solver.agv_arm.get_collision()
            C = np.concatenate(collision_center, axis=0)          # (N,3)
            R = np.concatenate(collision_r,  axis=0)[:, None]  # (N,1)

            all_C = np.concatenate((all_C, C), axis=0)
            all_R = np.concatenate((all_R, R), axis=0)
        


        #check collision
        dis_mat = distance.cdist(all_C, all_C)
        #check if any two points are closer than the sum of their radii
        radius_sum_mat = np.add.outer(all_R.reshape(-1), all_R.reshape(-1))


        remain_dis_mat = dis_mat - radius_sum_mat
        remain_dis_mat[np.diag_indices_from(remain_dis_mat)] = np.inf  # ignore self-distance

        if np.any(remain_dis_mat < 0):
            #collision detected
            return False
        else:
            #no collision
            return True

        
    def agv_ik_solver(self,grasp_pose,collision_points=None,vis_grasp_pose = True,select_flag = True):
        #grasp_pose: 4*4 matrix, x approach vector, y up vector, z normal vector, t grasp point
        #it should be turn into z approach vector, y up vector, x normal vector, t grasp point
        #return: q
        tool0_T = self.from_grasp_pose_to_tool0(grasp_pose)
        #change into from base link to tool0
        tool0_T[2,3] = tool0_T[2,3] - self.agv_base_height

        qs = self.ik_solver.solve(tool0_T,return_all_flag=True)


        # check collision with the scene
        CF_qs = []
        effort_list = []
        min_dis_list = []
        if collision_points is not None:
            cp = np.asarray(collision_points)           # (M,3)
            cp_kdtree = cKDTree(cp)


            for now_q in qs:
                #check collision with the scene
                T_fk = self.ik_solver.agv_arm.FK(now_q)
                collision_center,collision_r =  self.ik_solver.agv_arm.get_collision()
                C = np.concatenate(collision_center, axis=0)          # (N,3)
                R = np.concatenate(collision_r,  axis=0)[:, None]  # (N,1)

                #check self collision
                dis_mat = distance.cdist(C, C)
                #check if any two points are closer than the sum of their radii
                radius_sum_mat = np.add.outer(R.reshape(-1), R.reshape(-1))


                remain_dis_mat = dis_mat - radius_sum_mat
                remain_dis_mat[np.diag_indices_from(remain_dis_mat)] = np.inf  # ignore self-distance
                
                if np.any(remain_dis_mat < 0):
                    #collision detected
                    continue


                dists, _ = cp_kdtree.query(C, k=1, workers=1)  

                if np.all(dists > R):
                    CF_qs.append(now_q)
                    effort_list.append(np.sum(np.abs(now_q[3:])))
                    min_dis_list.append(np.min(dists))
        
        if len(CF_qs) == 0:
            print("no collision free solution")
            return None
        if select_flag:
            # select by min effort
            # choose_q_index = np.argmin(effort_list)
            choose_q_index = np.argmax(min_dis_list)
            choose_q = CF_qs[choose_q_index]
            merged_pc_o3d = o3d.geometry.PointCloud()
            # merged_pc_o3d.points = o3d.utility.Vector3dVector(self.merged_pc)
            merged_pc_o3d.points = o3d.utility.Vector3dVector(collision_points)

            

            #check joint limit
            lb = self.ik_solver.agv_arm.lower_bound[-6:]
            ub = self.ik_solver.agv_arm.upper_bound[-6:]

            period  = 2 * np.pi               
            angles  = choose_q[3:]            

            width       = ub - lb
            need_wrap   = (width < period) & ((angles < lb) | (angles > ub))

            if np.any(need_wrap):
                wrapped = ((angles[need_wrap] - lb[need_wrap]) % period) + lb[need_wrap]

                over_mask = wrapped > ub[need_wrap]
                wrapped[over_mask] -= period

                angles[need_wrap] = wrapped    

            choose_q[3:] = angles

            show_o3d = copy.deepcopy(merged_pc_o3d)
            show_points = np.array(show_o3d.points)
            show_points[:,2] -= self.agv_base_height
            show_o3d.points = o3d.utility.Vector3dVector(show_points)

            T_fk = self.ik_solver.agv_arm.FK(choose_q)
            if vis_grasp_pose:
                self.ik_solver.agv_arm.vis_collision(show_obj_list = [show_o3d],vis_coordinate=False)
            
            return choose_q
        else:
            # return all collision free solutions
            in_limit_qs = []
            for now_q in CF_qs:
                #check joint limit
                lb = self.ik_solver.agv_arm.lower_bound[-6:]
                ub = self.ik_solver.agv_arm.upper_bound[-6:]

                period  = 2 * np.pi               
                angles  = now_q[3:]            

                width       = ub - lb
                need_wrap   = (width < period) & ((angles < lb) | (angles > ub))

                if np.any(need_wrap):
                    wrapped = ((angles[need_wrap] - lb[need_wrap]) % period) + lb[need_wrap]

                    over_mask = wrapped > ub[need_wrap]
                    wrapped[over_mask] -= period

                    angles[need_wrap] = wrapped    

                now_q[3:] = angles

                in_limit_qs.append(now_q)
            return in_limit_qs


        
        

        
if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))   
    

    #read data for test
    camera_pose_list = []
    pc_list = []
    img_list = []
    
    file_name = "multi_imgs"
    img_index_list = [3]
    img_file_list = [os.path.join(current_path,f"../../example/{file_name}/robot1_rgb{i}.png") for i in img_index_list]
    pc_path_list = [os.path.join(current_path,f"../../example/{file_name}/chair_pc_{i-1}.npy") for i in img_index_list]
    camera_pose_file_list = [os.path.join(current_path,f"../../example/{file_name}/robot1_K_inv{i}.npy") for i in img_index_list]
    # img_file_list = [os.path.join(current_path,"../../example/multi_imgs/robot1_rgb1.png"),os.path.join(current_path,"../../example/multi_imgs/robot1_rgb2.png")]
    # pc_path_list = [os.path.join(current_path,"../../example/multi_imgs/table_pc_0.npy"),os.path.join(current_path,"../../example/multi_imgs/table_pc_1.npy")]
    # camera_pose_file_list = [os.path.join(current_path,"../../example/multi_imgs/robot1_K_inv1.npy"),os.path.join(current_path,"../../example/multi_imgs/robot1_K_inv2.npy")]

    # img_file_list = [os.path.join(current_path,"../../example/multi_imgs/robot1_rgb3.png"),os.path.join(current_path,"../../example/multi_imgs/robot1_rgb4.png"),os.path.join(current_path,"../../example/multi_imgs/robot1_rgb5.png")]
    # pc_path_list = [os.path.join(current_path,"../../example/multi_imgs/chair_pc_2.npy"),os.path.join(current_path,"../../example/multi_imgs/chair_pc_3.npy"),os.path.join(current_path,"../../example/multi_imgs/chair_pc_4.npy")]
    # camera_pose_file_list = [os.path.join(current_path,"../../example/multi_imgs/robot1_K_inv3.npy"),os.path.join(current_path,"../../example/multi_imgs/robot1_K_inv4.npy"),os.path.join(current_path,"../../example/multi_imgs/robot1_K_inv5.npy")]
    for img_path,pc_path,camera_pose_file in zip(img_file_list,pc_path_list,camera_pose_file_list):
        #load data
        pc = np.load(pc_path)
        camera_pose= np.load(camera_pose_file)
        now_img = np.array(Image.open(img_path)) 

        #to world frame
        world_pc = (camera_pose[0:3,0:3] @ pc.transpose()).T + camera_pose[0:3,3]

        keep_index = (world_pc[:,2]>0.01) & (world_pc[:,1] > -8)
        world_pc = world_pc[keep_index]

        camera_pose_list.append(camera_pose)
        pc_list.append(world_pc)
        img_list.append(now_img)

    #init grasp
    vlm_model = "qwen-vl-max-2025-04-08"
    potential_grasp_points = 10

    n_robot = 3
    large_obj_GP_generator = large_obj_grasping(img_list,pc_list,camera_pose_list,gripper_CAD_path=current_path,
                                                target_grasp_pose_num=potential_grasp_points,n_robot=n_robot,VLM_model=vlm_model)
    img_index = 0
    grasp_points = large_obj_GP_generator.generate_grasp_point(img_index)
    
    grasp_poses = large_obj_GP_generator.pose_generate(img_index,grasp_points,vis_grasp_flag=True)
    #save grasp poses
    # save_poses = np.array(grasp_poses)
    # save_path = os.path.join(current_path,"../experiments/compare_grasp_net/self_poses2.npy")
    # np.save(save_path, save_poses)

    now_instruction = "Move the table from A to B"
    # choose_index_list = large_obj_GP_generator.choose_grasp_pose(img_index,grasp_poses,now_instruction)
    choose_index_list = [0,1,2] #from human preference
    print("choose grasp poses")
    start = time.time()
    choose_grasp_pose = [grasp_poses[i] for i in choose_index_list]
    large_obj_GP_generator.collaborative_ik_solver(choose_grasp_pose,collision_points=large_obj_GP_generator.merged_pc,vis_grasp_pose=True)
    # for now_index in choose_index_list:
    #     print(now_index)
    #     print(grasp_poses[now_index])
    #     collision_points = large_obj_GP_generator.merged_pc
    #     grasp_q = large_obj_GP_generator.agv_ik_solver(grasp_poses[now_index],collision_points=collision_points)
    print("total time:",time.time()-start)

    # grasp_poses = large_obj_GP_generator.genera_feasible_grasp_poses(img_index,collision_points=large_obj_GP_generator.merged_pc,vis_grasp_flag=True)

