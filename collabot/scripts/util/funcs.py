from scipy.spatial.transform import Rotation as R
import numpy as np
import trimesh
import open3d as o3d
import os
import copy
from sklearn.cluster import DBSCAN

def quaternion_to_transform_matrix(quat, translation):
    
    rotation = R.from_quat(quat)
    
    rotation_matrix = rotation.as_matrix()
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix  
    transform_matrix[:3, 3] = translation       

    return transform_matrix

def sample_points(mesh_path, num_samples):
    mesh = trimesh.load(mesh_path)
    
    points = mesh.sample(num_samples)
    
    return points
import math
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def xyzrpy_to_T(q,xyz_first = True):
    if not xyz_first:
        q = [q[3],q[4],q[5],q[0],q[1],q[2]]
    T = np.eye(4)
    T[0:3, 3] = q[0:3]
    T[0:3, 0:3] = R.from_euler('xyz', q[3:]).as_matrix()
    return T

def T_to_xyzrpy(T,xyz_first = True):
    rpy = R.from_matrix(T[0:3, 0:3]).as_euler('xyz')
    xyz = T[0:3, 3]
    if xyz_first:
        return np.concatenate([xyz, rpy])
    else:
        return np.concatenate([rpy, xyz])
    
def path_inter(path,n_step = 100):
    new_path = []
    for i in range(1,len(path)):
        delta = path[i] - path[i-1]
        for j in range(n_step):
            new_path.append(path[i-1] + delta * j/n_step)
    new_path.append(path[-1])
    return new_path


def path_inter_total_points(path, total_pts=200):
    """
    Parameters
    ----------
    path : (N, d) array_like
        The original discrete path, including start and end points.
    total_pts : int
        The total number of sampled points in the new trajectory (including start and end). Minimum 2.

    Returns
    -------
    new_path : (total_pts, d) ndarray
        The trajectory after equidistant interpolation.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim == 1:           
        path = path[:, None]

    seg_vecs = np.diff(path, axis=0)            # (N-1,d)
    seg_len  = np.linalg.norm(seg_vecs, axis=1) # (N-1,)
    total_len = seg_len.sum()

    if total_pts < 2 or total_len == 0:
        return path.copy()         

    target_s = np.linspace(0.0, total_len, total_pts)  # (total_pts,)

    new_pts  = []
    seg_cum  = np.hstack(([0.0], np.cumsum(seg_len)))  # 0, l1, l1+l2, ...
    seg_idx  = 0

    for s in target_s:
        while seg_idx < len(seg_len) and s > seg_cum[seg_idx+1]:
            seg_idx += 1

        if seg_idx == len(seg_len):       
            new_pts.append(path[-1])
            continue

        s0, s1   = seg_cum[seg_idx], seg_cum[seg_idx+1]
        t        = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        pt       = path[seg_idx] + t * seg_vecs[seg_idx]
        new_pts.append(pt)

    return np.asarray(new_pts)

# def filter_pcd(pcd, nb_points=10, radius=0.5):
#     cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
#     filtered_pcd = pcd.select_by_index(ind)
#     return filtered_pcd

def filter_pcd(pcd,eps=0.5,min_sample = 20):
    points = np.array(pcd.points)
    # use DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_sample)
    labels = dbscan.fit_predict(points)

    unique_labels, counts = np.unique(labels, return_counts=True)
    # max_cluster_label = unique_labels[np.argmax(counts[unique_labels != -1])]
    #find max label except -1
    max_count = np.max(counts[unique_labels != -1])
    max_index = np.where(counts == max_count)[0]
    max_cluster_label = unique_labels[max_index]

    selected_index = np.where(labels == max_cluster_label)[0]
    filtered_pcd = pcd.select_by_index(selected_index)

    return filtered_pcd

def process_pcd(scene,obj_list,obj_center_list):
    #load the pcd file
    #world frame x y is the same as the gazebo, z is the same height as the robot base frame
    # print(obj_center_list)
    scene = filter_pcd(scene)
    #offset scene and table
    scene_points = np.array(scene.points)
    min_z = np.min(np.array(scene.points)[:,2])
    print(f"minimal height of the scene is {min_z}")

    scene_points[:,2] = scene_points[:,2] - min_z - 0.2
    scene.points = o3d.utility.Vector3dVector(scene_points)

    new_obj_list = []
    T_world_2_obj_list = []
    for now_obj,T_world_2_obj in zip(obj_list,obj_center_list):
        now_obj = filter_pcd(now_obj)
        #get the object center
        # object_center = now_obj.get_center()
        # T_world_2_obj = np.eye(4)
        # T_world_2_obj[0:2,3] = object_center[0:2]
        #motion planning for any object
        obj_points = np.array(copy.deepcopy(now_obj.points))
        obj_points[:,0] -= T_world_2_obj[0,3]
        obj_points[:,1] -= T_world_2_obj[1,3]
        obj_points[:,2] = obj_points[:,2] - T_world_2_obj[2,3] - 0.2

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(obj_points) #new object
        new_pcd.colors = now_obj.colors
        new_obj_list.append(new_pcd)
        T_world_2_obj_list.append(T_world_2_obj)

    return scene,new_obj_list,T_world_2_obj_list

def create_extrinsic_mat(obj_pcd,intrinsic_matrix,rot_degree = -45,img_width = 1920,img_height = 1080):
    #
    table_points = np.array(obj_pcd.points)
    table_center = obj_pcd.get_center()
    theta = rot_degree/180*np.pi #x-z plane
    z_axis = np.array([np.cos(theta),0,np.sin(theta)])
    x_axis = np.array([0,-1,0])
    y_axis = np.cross(z_axis,x_axis)

    R_ex_inv = np.column_stack((x_axis,y_axis,z_axis))

    #P uv = K R^T P - K R^T t_0 - alpha K R^T t = A + alpha B

    A_vecs = intrinsic_matrix @ R_ex_inv.T @ table_points.T - intrinsic_matrix @ R_ex_inv.T @ table_center.reshape(-1,1)

    B_vecs = - intrinsic_matrix @ R_ex_inv.T @ (-z_axis)

    left_bound_list = [] #alpha >= ?
    for i in range(A_vecs.shape[1]):
        now_a = A_vecs[:,i]
        now_b = B_vecs.reshape(-1)
        #consider x axis
        #use a alpha >= b form
        if now_b[0] > 0:
            left_bound_list.append(-now_a[0]/now_b[0])

        a_ = img_width * now_b[2] - now_b[0]
        b_ = now_a[0] - img_width * now_a[2]

        if a_ > 0:
            left_bound_list.append(b_/a_)
        
        #consider y axis
        if now_b[1] > 0:
            left_bound_list.append(-now_a[1]/now_b[1])

        a_ = img_height * now_b[2] - now_b[1]
        b_ = now_a[1] - img_height * now_a[2]
        if a_ > 0:
            left_bound_list.append(b_/a_)


    max_positive_value = np.max(np.array(left_bound_list)[np.array(left_bound_list) > 0])
    T_world_cam = np.eye(4)
    T_world_cam[0:3,0:3] = R_ex_inv
    T_world_cam[0:3,3] = table_center + max_positive_value * (-z_axis)

    extrinsic = np.linalg.inv(T_world_cam)

    return extrinsic

def generate_T_with_fixed_z(z_axis, N):
    angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
    z_axis = z_axis / np.linalg.norm(z_axis) 
    original_mat = R.from_euler('z',angles).as_matrix()

    dir = np.cross(np.array([0,0,1]), z_axis)
    dir = dir / np.linalg.norm(dir)
    angle = np.arccos(np.dot(np.array([0,0,1]), z_axis))
    rot_mat = R.from_rotvec(dir * angle).as_matrix()  

    result = np.einsum('jk,ikn->ijn',rot_mat,original_mat)
    return angles,result


def generate_ground_mesh(input_pc,ground_z):
    # Generate a ground mesh based on the input point cloud
    ground_z = np.min(input_pc[:, 2])
    min_x = np.min(input_pc[:, 0])
    max_x = np.max(input_pc[:, 0])
    min_y = np.min(input_pc[:, 1])
    max_y = np.max(input_pc[:, 1])

    # Create a grid for the ground mesh
    grid_res = 0.05  
    x_vals = np.arange(min_x, max_x, grid_res)
    y_vals = np.arange(min_y, max_y, grid_res)
    xx, yy = np.meshgrid(x_vals, y_vals)

    zz = np.full_like(xx, ground_z)

    ground_mesh = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    return ground_mesh