import numpy as np
import open3d as o3d
import copy
from sklearn.cluster import DBSCAN
import os 


def RigidTransform(R,t=np.zeros(3)):
    return np.array([[R[0,0],R[0,1],R[0,2],t[0]],
                     [R[1,0],R[1,1],R[1,2],t[1]],
                     [R[2,0],R[2,1],R[2,2],t[2]],
                     [0,0,0,1]])

def vis_grasp_pose(now_pc,grasp_pose,gripper_mesh):    
    tmp_pc = now_pc[0:3,:]
    norm_vector = now_pc[3:6,:]
    #using open3d to vis them
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp_pc.T)


    # data = np.load(npz_file_path)

    #load contact points
    # contact_point = now_contactpoints[i,0,:]


    #load grasp pose
    
    # grasp_pose = now_grasp_pose[i]
    # grasp_R = R.from_quat(grasp_pose[0:4]).as_matrix()
    # grasp_T = RigidTransform(grasp_R,grasp_pose[4:7])
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
    
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
    gripper_mesh_copy = copy.deepcopy(gripper_mesh)  # Copy the gripper mesh to avoid modifying the original
    gripper_mesh_copy.transform(grasp_pose)
    grasp_frame.transform(grasp_pose)


    # grasp_pose = gt_grasp_pose[i]

    # grasp_R = R.from_quat(gt_grasp_pose[0:4]).as_matrix()
    # grasp_T = RigidTransform(grasp_R,gt_grasp_pose[4:7])


    # Visualize the point cloud
    if gripper_mesh_copy is None:
        o3d.visualization.draw_geometries([pcd,grasp_frame,origin_frame])
    else:
        o3d.visualization.draw_geometries([pcd,gripper_mesh_copy,grasp_frame,origin_frame])

def vis_grasp_poses(now_pc, grasp_poses, gripper_mesh):    
    """
    Visualize multiple grasp poses on a point cloud using Open3D.
    
    Args:
        now_pc (np.ndarray): 6xN point cloud, first 3 rows are positions, next 3 are normals.
        grasp_poses (list or np.ndarray): List or array of N 4x4 grasp transformation matrices.
        gripper_mesh (o3d.geometry.TriangleMesh): Gripper mesh to visualize.
    """
    tmp_pc = now_pc[0:3, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp_pc.T)

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    vis_objects = [pcd, origin_frame]

    for grasp_pose in grasp_poses:
        # Add coordinate frame for each grasp pose
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        grasp_frame.transform(grasp_pose)

        # Add transformed gripper mesh
        gripper_mesh_copy = copy.deepcopy(gripper_mesh)
        gripper_mesh_copy.transform(grasp_pose)

        vis_objects.append(grasp_frame)
        vis_objects.append(gripper_mesh_copy)

    o3d.visualization.draw_geometries(vis_objects)
        

def load_ply_mesh(file_path,scale = 0.001,debug = True):

    mesh = o3d.io.read_triangle_mesh(file_path)  
    if not mesh.has_triangles():
        pcd = o3d.io.read_point_cloud(file_path)  
        # o3d.visualization.draw_geometries([pcd], window_name="PLY Point Cloud Visualization")
    else:
        mesh.compute_vertex_normals()

    mesh.scale(scale, center=[0,0,0])
    #move to origin
    center = mesh.get_center()

    mesh.translate(-center)

    offset = np.array([0, -0.1, 0])  # offset for the gripper
    mesh.translate(offset)
    rotation = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    transform = RigidTransform(rotation)
    mesh.transform(transform)
    if debug:
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
        o3d.visualization.draw_geometries([mesh,origin_frame], window_name="PLY Mesh Visualization")



    return mesh

def viewpoint_params_to_matrix(towards, angle):
    '''
    **Input:**

    - towards: numpy array towards vector with shape (3,).

    - angle: float of in-plane rotation.

    **Output:**

    - numpy array of the rotation matrix with shape (3, 3).
    '''
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)

def turn_o3d_into_input(pcd,contact_point,box_size = 0.5,target_points_num=2048,vis_flag = False):
    """
    this function is used to filter the point cloud and get the input for the network
    it will estimate the normal vector for the point cloud, which is necessary for pointnet++, not for pointnet
    """
    #first estimate the normal vector for the point cloud
    new_pcd = copy.deepcopy(pcd)
    new_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    nv = np.asarray(new_pcd.normals)
    point_cloud = np.asarray(new_pcd.points)


    point_cloud_now = point_cloud - contact_point

    max_try = 10
    for i in range(max_try):

        min_bound = np.zeros(3) - box_size / 2
        max_bound = np.zeros(3) + box_size / 2
        
        mask = ((point_cloud_now >= min_bound) & (point_cloud_now <= max_bound)).all(axis=1)
        filtered_points = point_cloud_now[mask]
        filtered_normals = nv[mask]
        box_size = box_size*1.5
        if len(filtered_points) > target_points_num:
            break
    

    
    if vis_flag:
        print(f"Before filter: {len(point_cloud_now)}, after filter: {len(filtered_points)}")
    
    
    np.random.seed(0) 
    indices = np.random.choice(len(filtered_points), size=target_points_num, replace=False)
    final_filtered_points = filtered_points[indices]
    final_filtered_normals = filtered_normals[indices]

    #vis the pc 
    if vis_flag:
        origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_filtered_points)
        pcd.normals = o3d.utility.Vector3dVector(final_filtered_normals)
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd,origin_axis])
    

    return np.concatenate((final_filtered_points, final_filtered_normals), axis=1).transpose()

#for eval
def filter_pcd(pcd,eps=0.5,min_sample = 20):
    points = np.array(pcd.points)
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
#for eval
def load_test_pcd(file_list):
    #load the pcd file
    #world frame x y is the same as the gazebo, z is the same height as the robot base frame
    pcd_dict = {}
    for obj_name in file_list:
        obj_path = os.path.join("PATH_to_CollaBot/", 'example/pc_data/{}.ply'.format(obj_name))
        print(obj_path)
        obj = o3d.io.read_point_cloud(obj_path)
        obj = filter_pcd(obj)
        if 'scene' in obj_name:
            obj = obj.voxel_down_sample(voxel_size=0.05)
        obj = filter_pcd(obj)
        pcd_dict[obj_name] = obj

    return pcd_dict
