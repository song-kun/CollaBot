import time

import numpy as np
from numpy import linalg
from math import pi, cos, sin, atan2, acos, sqrt, asin


# standard DH

DH_matrix_UR3e = np.asarray([[0, pi / 2.0, 0.15185],
                             [-0.24355, 0, 0],
                             [-0.2132, 0, 0],
                             [0, pi / 2.0, 0.13105],
                             [0, -pi / 2.0, 0.08535],
                             [0, 0, 0.0921]])

DH_matrix_UR5e = np.asarray([[0, pi / 2.0, 0.1625],
                             [-0.425, 0, 0],
                             [-0.3922, 0, 0],
                             [0, pi / 2.0, 0.1333],
                             [0, -pi / 2.0, 0.0997],
                             [0, 0, 0.0996]])

DH_matrix_UR10e = np.asarray([[0, pi / 2.0, 0.1807],
                              [-0.6127, 0, 0],
                              [-0.57155, 0, 0],
                              [0, pi / 2.0, 0.17415],
                              [0, -pi / 2.0, 0.11985],
                              [0, 0, 0.11655]])

DH_matrix_UR16e = np.asarray([[0, pi / 2.0, 0.1807],
                              [-0.4784, 0, 0],
                              [-0.36, 0, 0],
                              [0, pi / 2.0, 0.17415],
                              [0, -pi / 2.0, 0.11985],
                              [0, 0, 0.11655]])

DH_matrix_UR3 = np.asarray([[0, pi / 2.0, 0.1519],
                            [-0.24365, 0, 0],
                            [-0.21325, 0, 0],
                            [0, pi / 2.0, 0.11235],
                            [0, -pi / 2.0, 0.08535],
                            [0, 0, 0.0819]])

DH_matrix_UR5 = np.asarray([[0, pi / 2.0, 0.089159],
                            [-0.425, 0, 0],
                            [-0.39225, 0, 0],
                            [0, pi / 2.0, 0.10915],
                            [0, -pi / 2.0, 0.09465],
                            [0, 0, 0.0823]])

DH_matrix_UR10 = np.asarray([[0, pi / 2.0, 0.1273],
                             [-0.612, 0, 0],
                             [-0.5723, 0, 0],
                             [0, pi / 2.0, 0.163941],
                             [0, -pi / 2.0, 0.1157],
                             [0, 0, 0.0922]])


def get_DH_matrix_JAKA_ZU7_1200_ur():
    d1 = 0.12015
    d4 = 0.1135
    d5 = 0.1135
    d6 = 0.107
    a2 = -0.56
    a3 = -0.5035

    DH_matrix_JAKA_ZU7_1200_ur = np.asarray([[0, pi / 2.0, d1],
                                             [a2, 0, 0],
                                             [a3, 0, 0],
                                             [0, pi / 2.0, d4],
                                             [0, -pi / 2.0, d5],
                                             [0, 0, d6]])

    return DH_matrix_JAKA_ZU7_1200_ur



def get_DH_matrix_JAKA_ZU7_1200():
    d1 = 0.12015
    d4 = -0.1135
    d5 = 0.1135
    d6 = 0.107
    a2 = 0.56
    a3 = 0.5035

    DH_matrix_JAKA_ZU7_1200 = np.asarray([[0, pi / 2.0, d1],
                                          [a2, 0, 0],
                                          [a3, 0, 0],
                                          [0, pi / 2.0, d4],
                                          [0, -pi / 2.0, d5],
                                          [0, 0, d6]])

    return DH_matrix_JAKA_ZU7_1200

def get_DH_matrix_JAKA_ZU7():
    d1 = 0.12015
    a2 = 0.360
    a3 = 0.3035
    d4 = -0.11501
    d5 = 0.1135
    d6 = 0.107

    DH_matrix_JAKA_ZU7 = np.asarray([[0, pi / 2.0, d1],
                                          [a2, 0, 0],
                                          [a3, 0, 0],
                                          [0, pi / 2.0, d4],
                                          [0, -pi / 2.0, d5],
                                          [0, 0, d6]])

    return DH_matrix_JAKA_ZU7

def get_DH_matrix_JAKA_ZU7_ur():
    d1 = 0.12015
    a2 = -0.360
    a3 = -0.3035
    d4 = 0.11501
    d5 = 0.1135
    d6 = 0.107

    DH_matrix_JAKA_ZU7_1200_ur = np.asarray([[0, pi / 2.0, d1],
                                             [a2, 0, 0],
                                             [a3, 0, 0],
                                             [0, pi / 2.0, d4],
                                             [0, -pi / 2.0, d5],
                                             [0, 0, d6]])

    return DH_matrix_JAKA_ZU7_1200_ur

def inv_tran_matrix(trans):
    inv_trans = np.eye(4)
    inv_trans[:3, :3] = trans[:3, :3].transpose()
    inv_trans[:3, 3] = -inv_trans[:3, :3] @ trans[:3, 3]
    return inv_trans


def transform_dh(a, alpha, q, d):
    """
    **Input**

    - q:
    """
    trans = np.eye(4)
    c = np.cos(q)
    s = np.sin(q)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    ele_0_0 = c
    ele_0_1 = -s * c_alpha
    ele_0_2 = s * s_alpha
    ele_0_3 = a * c

    ele_1_0 = s
    ele_1_1 = c * c_alpha
    ele_1_2 = -c * s_alpha
    ele_1_3 = a * s

    ele_2_1 = s_alpha
    ele_2_2 = c_alpha
    ele_2_3 = d

    trans[0, 0] = ele_0_0
    trans[0, 1] = ele_0_1
    trans[0, 2] = ele_0_2
    trans[0, 3] = ele_0_3

    trans[1, 0] = ele_1_0
    trans[1, 1] = ele_1_1
    trans[1, 2] = ele_1_2
    trans[1, 3] = ele_1_3

    trans[2, 1] = ele_2_1
    trans[2, 2] = ele_2_2
    trans[2, 3] = ele_2_3

    return trans


def forward_kinematic_DH_solution(DH_matrix, edges):
    a = DH_matrix[:, 0]
    alpha = DH_matrix[:, 1]
    d = DH_matrix[:, 2]
    t01 = transform_dh(a[0], alpha[0], edges[0], d[0])
    t12 = transform_dh(a[1], alpha[1], edges[1], d[1])
    t23 = transform_dh(a[2], alpha[2], edges[2], d[2])
    t34 = transform_dh(a[3], alpha[3], edges[3], d[3])
    t45 = transform_dh(a[4], alpha[4], edges[4], d[4])
    t56 = transform_dh(a[5], alpha[5], edges[5], d[5])

    answer = t01 @ t12 @ t23 @ t34 @ t45 @ t56
    return answer


def inverse_kinematic_DH_solution(DH_matrix, transform_matrix):
    a = DH_matrix[:, 0]
    alpha = DH_matrix[:, 1]
    d = DH_matrix[:, 2]

    theta = np.zeros((6, 8))
    # theta 1
    T06 = transform_matrix

    # print(T06)
    # print(T06 @ np.array([[0], [0], [-DH_matrix[5, 2]], [1]]))
    P05 = T06 @ np.array([[0], [0], [-DH_matrix[5, 2]], [1]])
    # P05 = T06 @ np.array([0, 0, -DH_matrix[5, 2], 1])
    psi = atan2(P05[1], P05[0])

    phi = acos((DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2]) / sqrt(P05[0] ** 2 + P05[1] ** 2))
    theta[0, 0:4] = psi + phi + pi / 2
    theta[0, 4:8] = psi - phi + pi / 2

    # theta 5
    for i in [0, 4]:
        th5cos = (T06[0, 3] * sin(theta[0, i]) - T06[1, 3] * cos(theta[0, i]) - (
                DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2])) / DH_matrix[5, 2]
        if 1 >= th5cos >= -1:
            th5 = acos(th5cos)
        else:
            # print('invalid th5')
            th5 = 0
        theta[4, i:i + 2] = th5
        theta[4, i + 2:i + 4] = -th5

    # theta 6
    for i in [0, 2, 4, 6]:
        T60 = linalg.inv(T06)
        # T60 = inv_tran_matrix(T06)
        th5sin = sin(theta[4, i])
        # if abs(th5sin) < 0.000001:
        #     print('invalid th5')
        th = atan2((-T60[1, 0] * sin(theta[0, i]) + T60[1, 1] * cos(theta[0, i])) / th5sin,
                (T60[0, 0] * sin(theta[0, i]) - T60[0, 1] * cos(theta[0, i])) / th5sin)
        theta[5, i:i + 2] = th

    # theta 3
    for i in [0, 2, 4, 6]:
        T01 = transform_dh(a[0], alpha[0], theta[0, i], d[0])
        T45 = transform_dh(a[4], alpha[4], theta[4, i], d[4])
        T56 = transform_dh(a[5], alpha[5], theta[5, i], d[5])
        T14 = linalg.inv(T01) @ T06 @ linalg.inv(T45 @ T56)
        P13 = T14 @ np.array([[0], [-DH_matrix[3, 2]], [0], [1]])
        th3cos = ((P13[0] ** 2 + P13[1] ** 2 - DH_matrix[1, 0] ** 2 - DH_matrix[2, 0] ** 2) /
                (2 * DH_matrix[1, 0] * DH_matrix[2, 0]))
        if 1 >= th3cos >= -1:
            th3 = acos(th3cos)
        else:
            # print('invalid th3')
            th3 = 0
        theta[2, i] = th3
        theta[2, i + 1] = -th3

    # theta 2,4
    for i in range(8):
        T01 = transform_dh(a[0], alpha[0], theta[0, i], d[0])
        T45 = transform_dh(a[4], alpha[4], theta[4, i], d[4])
        T56 = transform_dh(a[5], alpha[5], theta[5, i], d[5])
        T14 = linalg.inv(T01) @ T06 @ linalg.inv(T45 @ T56)
        P13 = T14 @ np.array([[0], [-DH_matrix[3, 2]], [0], [1]])

        theta[1, i] = atan2(-P13[1], -P13[0]) - asin(
            -DH_matrix[2, 0] * sin(theta[2, i]) / sqrt(P13[0] ** 2 + P13[1] ** 2)
        )
        T32 = linalg.inv(transform_dh(a[2], alpha[2], theta[2, i], d[2]))
        T21 = linalg.inv(transform_dh(a[1], alpha[1], theta[1, i], d[1]))
        T34 = T32 @ T21 @ T14
        theta[3, i] = atan2(T34[1, 0], T34[0, 0])

    return theta
    



def nearestQ(q_list, last_q):
    """
    Function that computes the distance from every new configuration to the previous
    :param q_list: np array (6, 8)
    :param last_q: previous configuration, np array (6, )
    :return: closest configuration to the previous
    """
    weights = np.array([6, 5, 4, 3, 2, 1])

    diff = q_list - last_q[:, np.newaxis]  # Broadcasting last_q to shape (6, 8)
    weighted_diff = diff * weights[:, np.newaxis]  # Broadcasting weights to shape (6, 8)
    conf_dists = np.sum(weighted_diff ** 2, axis=0)

    min_index = np.argmin(conf_dists)
    best_q = q_list[:, min_index]

    return best_q


def convert_jaka_Transform2ur(T_base_end_J, Jaka_dh_param):
    """
    Convert
        the transformation matrix corresponding to the JAKA standard DH coordinate frame
        to
        the transformation matrix corresponding to the UR standard DH coordinate frame
    For Jaka inverse kinematics

    :param T_base_end_J: np array (4, 4), transform matrix corresponding to the JAKA standard DH coordinate frame
    :param Jaka_dh_param: np array (4, 4), JAKA standard DH parameters

    :return: T_base_end_U: np array (4, 4), transform matrix corresponding to the UR standard DH coordinate frame
    """
    d6 = Jaka_dh_param[5, 2]
    T_base_U_J = np.asarray([[-1, 0, 0, 0.],
                             [0, -1, 0, 0.],
                             [0, 0, 1, 0.],
                             [0, 0, 0, 1]])
    T_base_J_U = np.asarray([[-1, 0, 0, 0.],
                             [0, -1, 0, 0.],
                             [0, 0, 1, 0.],
                             [0, 0, 0, 1]])

    T_end_U_J = np.asarray([[-1, 0, 0, 0.],
                            [0, 1, 0, 0.],
                            [0, 0, -1, -2 * d6],
                            [0, 0, 0, 1]])
    T_end_J_U = np.asarray([[-1, 0, 0, 0.],
                            [0, 1, 0, 0.],
                            [0, 0, -1, -2 * d6],
                            [0, 0, 0, 1]])

    # T_base_U_J @ T_base_end_J: T_base_U_base_J @ T_base_J_end_J = T_base_U_end_J
    # T_base_U_end_J @ T_end_J_U: T_base_U_end_J @ T_end_J_end_U = T_base_U_end_U = T_base_end_U
    T_base_end_U = T_base_U_J @ T_base_end_J @ T_end_J_U

    return T_base_end_U


def JAKA_inverse_kinematic_DH_solution(jaka_DH_matrix, jaka_DH_matrix_ur, transform_matrix):
    """
    Jaka inverse kinematics

    :param jaka_DH_matrix: JAKA standard DH parameters: in JAKA format
    :param jaka_DH_matrix_ur: JAKA standard DH parameters: in UR format
    :param transform_matrix: np array (4, 4), transform matrix corresponding to the JAKA standard DH coordinate frame

    :return: np array (6, 8), inverse kinematics result
    """
    try:
        T_base_end_U = convert_jaka_Transform2ur(transform_matrix, jaka_DH_matrix)
        IKS_res = inverse_kinematic_DH_solution(jaka_DH_matrix_ur, T_base_end_U)
        IKS_res[[1, 2, 3, 5], :] *= -1  # (6, 8)
        return IKS_res
    except:
        return np.array([])


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)

    # ed = np.asarray([[1.572584629058838], [-1.566467599277832], [-0.0026149749755859375], [-1.568673924808838],
    #                  [-0.009446446095601857], [0.007950782775878906]])
    # ed = np.asarray([1.572584629058838, -1.566467599277832, -0.0026149749755859375, -1.568673924808838,
    #                  -0.009446446095601857, 0.007950782775878906])

    ed = np.asarray([3.315783, 2.240158, -2.179694, 1.983942, 1.634298, 1.492484])
    # ed = np.asarray([[3.315783], [2.240158], [-2.179694], [1.983942],
    #                 [1.634298], [1.492484]])
    # ed = np.asarray([0,0,0,0,0,0])

    DH_matrix_JAKA_ZU7_1200 = get_DH_matrix_JAKA_ZU7_1200()
    T_0_6J = forward_kinematic_DH_solution(DH_matrix_JAKA_ZU7_1200, ed)
    print("Forward")
    print(T_0_6J)

    print("Inverse")
    DH_matrix_JAKA_ZU7_1200_ur = get_DH_matrix_JAKA_ZU7_1200_ur()
    # t1 = time.time()
    T_0_6U = convert_jaka_Transform2ur(T_0_6J, DH_matrix_JAKA_ZU7_1200)
    IKS = inverse_kinematic_DH_solution(DH_matrix_JAKA_ZU7_1200_ur, T_0_6U)
    IKS[[1, 2, 3, 5], :] *= -1  # (6, 8)
    # print((time.time()-t1)*1000)

    IKS = JAKA_inverse_kinematic_DH_solution(DH_matrix_JAKA_ZU7_1200, DH_matrix_JAKA_ZU7_1200_ur, T_0_6J)

    for i in range(8):
        print('%d: ' % i, IKS[:, i])
    print('===========')
    print(T_0_6J)
    for i in range(8):
        transform = forward_kinematic_DH_solution(DH_matrix_JAKA_ZU7_1200, IKS[:, i])
        print(i, '----')
        print(transform-T_0_6J)