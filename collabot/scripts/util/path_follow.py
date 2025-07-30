import numpy as np
import matplotlib.pyplot as plt
from util.funcs import angle_normalize,xyzrpy_to_T,T_to_xyzrpy
import time
from scipy.spatial.transform import Rotation as R


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

class simple_PID_control:
    def __init__(self,dim,Kp = 1,Ki = 1,Kd = 1):
        self.dim = dim
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0
        self.error_last = np.zeros(self.dim)
        self.error_sum = np.zeros(self.dim)

    def control(self,now_error):
        self.error = now_error
        self.error_sum += self.error
        output = self.Kp * self.error + self.Ki * self.error_sum + self.Kd * (self.error - self.error_last)
        self.error_last = self.error

        return output
import matplotlib.animation as animation
from IPython.display import HTML

# 你的 animate_trajectories 函数和调用代码


def animate_trajectories(a, b,label_list = ["a","b"]):
    N = a.shape[0]  # Assumes a and b are both Nx2 arrays

    fig, ax = plt.subplots()
    line_a, = ax.plot([], [], 'r-',label=label_list[0])  # Red line for trajectory a
    line_b, = ax.plot([], [], 'b-',label = label_list[1])  # Blue line for trajectory b

    # Setting the limits for x and y axes to ensure all points are visible
    small_num = 0.1
    ax.set_xlim(np.min(np.concatenate([a[:,0], b[:,0]])) -small_num, np.max(np.concatenate([a[:,0], b[:,0]])) +small_num)
    ax.set_ylim(np.min(np.concatenate([a[:,1], b[:,1]]))-small_num, np.max(np.concatenate([a[:,1], b[:,1]])) +small_num)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # ax.set_title(f"Trajectories of robot {robot_index+1}")
    #set equal
    ax.set_aspect('equal', 'box')
    #legend
    ax.legend()

    # Initialize function for the animation
    def init():
        line_a.set_data([], [])
        line_b.set_data([], [])
        return line_a, line_b,

    # Animation function which updates the plot
    def animate(i):
        line_a.set_data(a[:i+1, 0], a[:i+1, 1])
        line_b.set_data(b[:i+1, 0], b[:i+1, 1])
        return line_a, line_b,

    # Creating the animation using the FuncAnimation module
    ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, blit=True, repeat=False, interval=200)

    plt.close()
    return ani

class path_following:
    def __init__(self,robot_list,move_pid = [6,0.3,4], rot_pid = [6,0.5,0.5]):
        self.robot_name = robot_list
        self.dim = len(robot_list)

        self.move_pid = move_pid
        self.rot_pid = rot_pid
        self.max_speed = np.array([0.5,0.5,0.5])

        self.real_pose_list = None
        self.target_pose = None
        self.object_pose_inter = 10

    
    def create_PID_controller(self):
        self.move_controller = [simple_PID_control(2,self.move_pid[0],self.move_pid[1],self.move_pid[2]) for i in range(self.dim)]
        self.rot_controller = [simple_PID_control(1,self.rot_pid[0],self.rot_pid[1],self.rot_pid[2]) for i in range(self.dim)]


    def follow_path(self,robot_pose_list,swarm_control,get_pose_func,obj_pose_list = None, assign_obj_pose_func=None,obj_name = None, dt = 0.5):
        #obj_pose_list:[xyzrpy]
        self.target_pose = robot_pose_list
        self.create_PID_controller()
        real_pose_list = []
        pid_list = self.move_controller
        w_pid_list = self.rot_controller

        for i in range(len(robot_pose_list) - 1):
            
            next_pose = robot_pose_list[i+1]
            now_pose = get_pose_func()
            real_pose_list.append(now_pose)
            for name_,now_pose_,next_pose_,pid_,w_pid_ in zip(self.robot_name,now_pose,next_pose,pid_list,w_pid_list):
                
                now_error = (next_pose_[0:2] - now_pose_[0:2])

                rot_error = angle_normalize(next_pose_[2] - now_pose_[2])

                control_output = pid_.control(now_error)
                rot_control_output = w_pid_.control(rot_error)

                #change into now_pose_frame
                roted_xy = R.from_euler('z',-now_pose_[2]).as_matrix()[0:2,0:2] @ control_output[0:2]
                agv_velocity = np.array([roted_xy[0],roted_xy[1],rot_control_output[0]])
                agv_velocity = np.clip(agv_velocity,-self.max_speed,self.max_speed)

                swarm_control.base_v(name_,agv_velocity)
                if len(next_pose_) == 9:
                    swarm_control.arm_pose(name_,next_pose_[3:])
            
            if obj_pose_list is not None:
                T_list = interpolation_T(xyzrpy_to_T(obj_pose_list[i]),xyzrpy_to_T(obj_pose_list[i+1]),self.object_pose_inter)
                for next_T in T_list:
                    next_pose = next_T[:3,3]
                    quat = R.from_matrix(next_T[:3,:3]).as_quat()
                    assign_obj_pose_func(obj_name,next_pose,quat)
                    time.sleep(dt/self.object_pose_inter)
            else:
                time.sleep(dt)
            
        now_pose = get_pose_func()
        real_pose_list.append(now_pose)
        self.real_pose_list = real_pose_list

        for name_ in self.robot_name:
            swarm_control.base_v(name_,[0,0,0])
            # swarm_control.arm_pose(name_,[0,0,0])
        print("path following finished")
    
    def visual_last_error(self):
        real_pose_list = np.array(self.real_pose_list)
        target_pose = np.array(self.target_pose)

        for robot_index in range(len(self.robot_name)):
            error = real_pose_list[:,robot_index,0:3]-target_pose[:,robot_index,0:3]
            error[:,2] = angle_normalize(error[:,2])
            error_norm = np.linalg.norm(error,axis = 1)
            plt.plot(error_norm,label = f"robot{robot_index + 1}")
        # plt.axis('equal')
        plt.legend()
        plt.show()
    
    def visusal_last_dim(self,robot_index,dim):
        real_pose_list = np.array(self.real_pose_list)
        targe_pose = np.array(self.target_pose)
        real_pose_list[:,robot_index,2] = angle_normalize(real_pose_list[:,robot_index,2])
        targe_pose[:,robot_index,2] = angle_normalize(targe_pose[:,robot_index,2])

        # error = real_pose_list[:,robot_index,dim]-targe_pose[:,robot_index,dim]
        plt.plot(real_pose_list[:,robot_index,dim],label = "real")
        plt.plot(targe_pose[:,robot_index,dim],label = "target")
        
        plt.show

    def visual_last_trajectory(self,robot_index):
        real_pose_list = np.array(self.real_pose_list)
        target_pose = np.array(self.target_pose)

        # 调用函数，并将输出作为 HTML 显示
        ani = animate_trajectories( real_pose_list[:,robot_index,0:3], target_pose[:,robot_index,0:3],['real pose','target pose'])
        return ani
        
                
        