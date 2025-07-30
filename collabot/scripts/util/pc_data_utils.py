""" Tools for data processing.
"""

import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt

class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

def vis_points(points, color, mask = None):
    #points: Nx3 or HxWx3
    #color: Nx3 or HxWx3
    if mask is not None:
        index = mask == 1
        points = points[index]
        color = color[index]
    points = points.reshape(-1,3)
    color = color.reshape(-1,3)
    color = color/255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
    o3d.visualization.draw_geometries([pcd])

def merge_clouds(clouds, camera_K_list):
    #clouds: [cloud1, cloud2,.....], each cloud is a N*3 array
    # each clouds in camera frame
    #camera_K_list: [K1, K2,....], each K is a 4*4 array
    total_clouds = np.array([]).reshape(-1,3)

    for now_cloud, now_K in zip(clouds, camera_K_list):
        k_inv = np.linalg.inv(now_K)
        now_cloud =k_inv[0:3,0:3] @ now_cloud.T 
        now_cloud = now_cloud.T + k_inv[0:3,3]
        total_clouds = np.concatenate((total_clouds, now_cloud), axis=0)
    return total_clouds

def down_sample(points, color=None, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if color is not None:
        return np.array(downpcd.points), np.array(downpcd.colors)
    else:
        return np.array(downpcd.points)

    
def mask_to_pc(mask, cloud,color):
    #mask: the mask of the object
    #cloud: the point cloud of the object
    #return: the point cloud of the object with mask
    mask = mask.flatten()
    cloud = cloud.reshape(-1,3)
    pc = cloud[mask==1]
    color = color.reshape(-1,3)
    color = color[mask==1]
    return pc,color

# def remove_obj_from_scene(scene_pc, scene_color, obj_pc):
#     #scene_pc: pcd
#     #scene_color: pcd

#     obj_pcd = o3d.geometry.PointCloud()
#     obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)  # 设置点

#     kdtree = o3d.geometry.KDTreeFlann(obj_pcd)

#     # 要删除的点的索引
#     to_delete = []

#     # 检查点云A中的每个点
#     for i, point in enumerate(scene_pc):
#         # 搜索点云B中最近的点
#         [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
#         if k > 0:
#             dist = np.linalg.norm(np.asarray(obj_pc)[idx[0]] - np.asarray(point))
#             # 如果距离小于一个很小的阈值，认为是同一个点
#             if dist < 0.1:
#                 to_delete.append(i)

#     # 从点云A中删除这些点
#     scene_pc = np.delete(scene_pc, to_delete, axis=0)
#     scene_color = np.delete(scene_color, to_delete, axis=0)

#     return scene_pc, scene_color

def remove_obj_from_scene(scene_pcd, obj_pcd):
    #scene_pc: pcd
    #scene_color: pcd

    kdtree = o3d.geometry.KDTreeFlann(obj_pcd)

    # 要删除的点的索引
    to_delete = []

    # 检查点云A中的每个点
    for i, point in enumerate(scene_pcd.points):
        # 搜索点云B中最近的点
        [k, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        if k > 0:
            dist = np.linalg.norm(np.asarray(obj_pcd.points)[idx[0]] - np.asarray(point))
            # 如果距离小于一个很小的阈值，认为是同一个点
            if dist < 0.1:
                to_delete.append(i)

    # 从点云A中删除这些点
    scene_pcd.points = o3d.utility.Vector3dVector(np.delete(np.asarray(scene_pcd.points), to_delete, axis=0))
    scene_pcd.colors = o3d.utility.Vector3dVector(np.delete(np.asarray(scene_pcd.colors), to_delete, axis=0))


    return scene_pcd

def write_pc(path,points,color):
    #points: N*3
    #color: N*3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(path, pcd)

def load_pc(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    color = np.asarray(pcd.colors)
    return points, color

def to_o3d_pcd(points, color, normalized_color = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if not normalized_color:
        color = color/255.0
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def remove_ground(scene_pc, scene_color, obj_pc, obj_color):
    #remove the ground plane in obj

    ground_z = np.min(scene_pc[:,2])

    ground_index = np.where(obj_pc[:,2] < ground_z + 0.05)

    ground_pc_in_obj = obj_pc[ground_index]
    ground_color_in_obj = obj_color[ground_index]

    obj_pc = np.delete(obj_pc, ground_index, axis=0)
    obj_color = np.delete(obj_color, ground_index, axis=0)

    #add in the scene
    scene_pc = np.concatenate((scene_pc, ground_pc_in_obj), axis=0)
    scene_color = np.concatenate((scene_color, ground_color_in_obj), axis=0)

    return scene_pc, scene_color, obj_pc, obj_color

def simple_remove_ground(pc):
    #remove the ground plane in obj
    #pc: N*3
    #remove the points below the z axis
    ground_z = np.min(pc[:,2])
    ground_index = np.where(pc[:,2] < ground_z + 0.05)
    pc = np.delete(pc, ground_index, axis=0)
    return pc

def compute_curvature(input_pcd,vis_flag = False):
    # 估计点云的法线
    pcd = copy.deepcopy(input_pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 计算曲率
    curvatures = []
    for p in np.asarray(pcd.points):
        # 获取K最近邻点
        
        k = 10  # 选择邻点数量
        _, idx, _ = kdtree.search_knn_vector_3d(p, k)
        
        # 从邻点构建局部点云
        neighbour_pcd = pcd.select_by_index(idx)
        
        # 计算邻点协方差矩阵
        cov = np.cov(np.asarray(neighbour_pcd.points).T)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # 曲率估计为最小特征值除以所有特征值之和
        curvature = eigenvalues[0] / sum(eigenvalues)
        curvatures.append(curvature)

    # 将曲率作为点云的一个属性
    pcd_curvatures = np.array(curvatures)
    if vis_flag:
        print(f"Curvatures computed. find {len(pcd_curvatures)} points")

        # 可视化点云的曲率，可以映射曲率到颜色
        colors = plt.get_cmap("viridis")(pcd_curvatures / np.max(pcd_curvatures))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    return pcd_curvatures


def show_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

from sklearn.cluster import DBSCAN

def remove_outlier(points,eps=0.5,min_sample = 20):
    dbscan = DBSCAN(eps=eps, min_samples=min_sample)
    labels = dbscan.fit_predict(points)

    unique_labels, counts = np.unique(labels, return_counts=True)
    # max_cluster_label = unique_labels[np.argmax(counts[unique_labels != -1])]
    #find max label except -1
    max_count = np.max(counts[unique_labels != -1])
    max_index = np.where(counts == max_count)[0]
    max_cluster_label = unique_labels[max_index]

    selected_index = np.where(labels == max_cluster_label)[0]
    filtered_points = points[selected_index]

    return filtered_points

from sklearn.cluster import KMeans

def extract_cluster_centers(all_grasp_points, M):
    """
    Extract M cluster centers from Nx3 point cloud using KMeans.
    
    Args:
        all_grasp_points (np.ndarray): Nx3 point cloud array.
        M (int): Number of cluster centers to extract.
    
    Returns:
        np.ndarray: Mx3 array of cluster centers.
    """
    kmeans = KMeans(n_clusters=M, random_state=0,n_init='auto')
    kmeans.fit(all_grasp_points)
    centers = kmeans.cluster_centers_
    return centers

from scipy.spatial import cKDTree

def compute_nearest_distances(B_points, A_points):
    """
    Compute the shortest distance from each point in B to the nearest point in A.

    Args:
        B_points (np.ndarray): An (n, 3) array representing the point cloud B with n points.
        A_points (np.ndarray): An (m, 3) array representing the point cloud A with m points.

    Returns:
        np.ndarray: An (n,) array containing the nearest distance from each point in B to A.
    """
    tree = cKDTree(A_points)
    distances, _ = tree.query(B_points, k=1)  # k=1 means only the nearest neighbor is queried
    return distances