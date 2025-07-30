import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R



def draw_axis_on_img(image, extrinsics, intrinsics, thickness=3,line_length = 1):

    # the points in the 3D space
    points_3d = np.float32([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]]).reshape(-1, 3)
    r_vec = R.from_matrix(extrinsics[0:3, 0:3]).as_rotvec()
    t_vec = extrinsics[0:3, 3]

    # project the points into the image
    points_2d, _ = cv2.projectPoints(points_3d, r_vec, t_vec, intrinsics, np.zeros(5))
    points_2d = points_2d.astype(np.int32)
    # draw the axis
    image = cv2.arrowedLine(image, tuple(points_2d[0].ravel()), tuple(points_2d[1].ravel()), (255, 0, 0), thickness)
    image = cv2.arrowedLine(image, tuple(points_2d[0].ravel()), tuple(points_2d[2].ravel()), (0, 255, 0), thickness)
    image = cv2.arrowedLine(image, tuple(points_2d[0].ravel()), tuple(points_2d[3].ravel()), (0, 0, 255), thickness)
    return image
