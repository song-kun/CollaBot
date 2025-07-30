import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))

from SEEM.seem_main import SEEM
from util.pc_data_utils import CameraInfo, create_point_cloud_from_depth_image,mask_to_pc
import numpy as np
from scipy.spatial.transform import Rotation as R

from PIL import Image


class objects_seg:
    def __init__(self, colors, depths, camera,target_objs):
        #colors: [img1, img2, ....]
        #depths: [img1, img2,.....]
        #camera: camera info

        self.colors = colors
        self.depths = depths
        self.camera = camera
        self.target_objs = target_objs
        self.seem = SEEM()
        self.obj_img_dict = {} #{obj1: [img1_index,img2_index,....]}
    
    def get_mask(self, obj):
        obj_imgs = [self.colors[i] for i in self.obj_img_dict[obj]]
        mask_list = []
        for img in obj_imgs:
            masks,result = self.seem.seg_img_with_text(img,obj)
            if len(masks) > 0:
                mask_list.append(masks[0])
            else:
                mask_list.append(np.zeros_like(img))
        return mask_list
    
    def get_obj_cloud(self, obj):
        obj_depth_img = [self.depths[i] for i in self.obj_img_dict[obj]]
        # obj_color_img = self.colors[self.obj_img_dict[obj]]
        clouds = []
        for i in range(len(obj_depth_img)):
            cloud = create_point_cloud_from_depth_image(obj_depth_img[i], self.camera, organized=True)
            clouds.append(cloud)
        obj_mask = self.get_mask(obj)
        return clouds, obj_mask


if __name__ == '__main__':
    from pc_data_utils import vis_points,merge_clouds

    current_path = os.path.dirname(os.path.abspath(__file__))
    intri_path = os.path.join(current_path,'../../param/intrinsics.npy')
    intrinsic = np.load(intri_path)
    factor_depth =  1000
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    #load the img
    color_img_list = []
    depth_img_list = []
    K_list = []

    
    data_dir = os.path.join(current_path, '../../example/multi_imgs')
    for i in range(1,6):
        color = np.array(Image.open(os.path.join(data_dir, f'robot1_rgb{i}.png'))) 
        depth = np.array(Image.open(os.path.join(data_dir, f'robot1_depth{i}.png')))
        K_inv = np.load(os.path.join(data_dir, f'robot1_K_inv{i}.npy'))
        color_img_list.append(color)
        depth_img_list.append(depth)
        K_list.append(np.linalg.inv(K_inv))
    

    now_seg = objects_seg(color_img_list,depth_img_list,camera,['table','chair'])

    #get the mask of the object
    now_seg.obj_img_dict['table'] = [0,1]
    now_seg.obj_img_dict['chair'] = [2,3,4]

    target_obj = 'chair'
    clouds, masks = now_seg.get_obj_cloud(target_obj)

    #merge the clouds
    new_clouds = []
    new_color = np.array([]).reshape(-1,3)
    for index, (cloud, mask) in enumerate(zip(clouds, masks)):
        img_index = now_seg.obj_img_dict[target_obj][index]
        pc,color = mask_to_pc(mask, cloud, color_img_list[img_index])
        vis_points(pc, color)
        new_clouds.append(pc)
        new_color = np.concatenate((new_color, color), axis=0)
    obj_K_list = [K_list[i] for i in now_seg.obj_img_dict[target_obj]]
    total_cloud = merge_clouds(new_clouds, obj_K_list)

    #vis
    vis_points(total_cloud, new_color)


