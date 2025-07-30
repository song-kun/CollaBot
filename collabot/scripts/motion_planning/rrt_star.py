"""
RRT star with constrain
"""

import math
import random

import matplotlib.pyplot as plt
import torch
import numpy as np
def angle_normalize(x):
    #use torch
    return torch.fmod(x + torch.pi, 2*torch.pi) - torch.pi

def random_sample(min_state, max_state, dim):
    """

    Args:
        min_state: tensor
        max_state: tensor
        dim: int

    Returns:

    """
    random_tensor = torch.rand(dim)
    rnd = random_tensor * (max_state - min_state) + min_state
    return rnd

class RRTStar:
    class Node:
        def __init__(self, state):
            """

            Args:
                state: tensor type
            """
            self.state = state
            self.path = None  # a state tensor from parent to node
            self.parent = None
            self.cost = 0.0

    def __init__(
        self,
        start=None,
        goal=None,
        dim=2,
        min_state=None,
        max_state=None,
        collision_check=None,
        expand_dis=30.0,
        path_resolution=0.1,
        goal_sample_rate=10,
        max_iter=300,
        connect_circle_dist=50.0,
        rotation_index = []
    ):
        """

        Args:
            start:
            dim:
            min_state:
            max_state:
            collision_check:
            expand_dis: 更新节点距离
            path_resolution: 节点间的碰撞检查间隔
            goal_sample_rate: 目标点采样频率
            max_iter:
            connect_circle_dist: 检查节点范围
            goal:
        """
        self.dim = dim
        self.start = self.Node(start.float())
        if goal is None:
            self.end = self.Node(start.float())
        else:
            self.end = self.Node(goal.float())

        self.min_rand = min_state.float()
        self.max_rand = max_state.float()

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.collision_check = collision_check
        self.rotation_index = rotation_index

        self.node_list = [self.start]
        self.node_state_list = self.start.state

    def set_goal(self, state):
        """

        Args:
            state: tensor

        Returns:

        """
        self.end = self.Node(state.float())

    def planning(self,debug = False):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        for i in range(self.max_iter):
            if debug:
                print("Iter:", i, ", number of nodes:", len(self.node_list))

            rnd = self.get_random_node()

            distances = self.cal_state_dist_batch(
                self.node_state_list, rnd.state.view(1, -1)
            )
            nearest_ind = torch.argmin(distances).item()
            near_node = self.node_list[nearest_ind]

            new_node = self.steer(near_node, rnd, self.expand_dis)
            new_node.cost = near_node.cost + self.cal_state_dist(
                near_node.state, new_node.state
            )

            if self.collision_check(new_node.state):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)

                self.rewire(new_node, near_inds)
                self.node_list.append(new_node)
                self.node_state_list = torch.cat(
                    [self.node_state_list, new_node.state], dim=0
                )

                if (
                    self.cal_state_dist(new_node.state, self.end.state).item()
                    < self.expand_dis
                ):
                    last_index = self.search_best_goal_node()
                    if last_index is not None:
                        return self.generate_final_course(last_index)
            if debug:
                self.draw_graph(rnd)
        if debug:
            print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            random_tensor = torch.rand(self.dim)
            rnd = random_tensor * (self.max_rand - self.min_rand) + self.min_rand
            rnd = self.Node(rnd.unsqueeze(0))
        else:
            rnd = self.Node(self.end.state)
        return rnd

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.state)
        new_node.path = new_node.state

        d_state = to_node.state - from_node.state
        d_state[0,self.rotation_index] = angle_normalize(d_state[0,self.rotation_index])


        d = torch.norm(d_state)
        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        d_state = self.path_resolution / d * d_state

        for _ in range(n_expand):
            new_node.state = new_node.state + d_state
            new_node.state[0,self.rotation_index] = angle_normalize(new_node.state[0,self.rotation_index])

            new_node.path = torch.cat([new_node.path, new_node.state], dim=0)

        d = self.cal_state_dist(new_node.state, to_node.state)
        if d <= self.path_resolution:
            new_node.state = to_node.state
            new_node.path = torch.cat([new_node.path, new_node.state], dim=0)

        new_node.parent = from_node

        return new_node

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the tree that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        r = self.connect_circle_dist

        distances = self.cal_state_dist_batch(
            self.node_state_list, new_node.state.view(1, -1)
        )

        near_inds = [i for i in range(distances.size(0)) if distances[i] <= r]

        return near_inds

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return new_node

        # search nearest cost in near_inds
        costs = []
        t_node = []
        j = 0

        for i in near_inds:
            near_node = self.node_list[i]
            t_node.append(self.steer(near_node, new_node))
            if t_node[j] and self.collision_check(t_node[j].state):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
            j = j + 1

        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return new_node

        new_node = t_node[costs.index(min_cost)]
        new_node.cost = min_cost

        return new_node

    def calc_new_cost(self, from_node, to_node):
        d = self.cal_state_dist(from_node.state, to_node.state)
        return from_node.cost + d

    def rewire(self, new_node, near_inds):
        """
        For each node in near_inds, this will check if it is cheaper to
        arrive to them from new_node.
        In such a case, this will re-assign the parent of the nodes in
        near_inds to new_node.
        Parameters:
        ----------
            new_node, Node
                Node randomly added which can be joined to the tree

            near_inds, list of uints
                A list of indices of the self.new_node which contains
                nodes within a circle of a given radius.
        Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            new_cost = self.calc_new_cost(new_node, near_node)

            if near_node.cost > new_cost:
                near_node.parent = new_node

    def search_best_goal_node(self):
        dist_to_goal_list = self.cal_state_dist_batch(
            self.node_state_list, self.end.state.view(1, -1)
        )
        goal_inds = [
            i
            for i in range(dist_to_goal_list.size(0))
            if dist_to_goal_list[i] <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.end)
            if self.collision_check(t_node.state):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [
            self.node_list[i].cost
            + self.cal_state_dist(self.node_list[i].state, self.end.state)
            for i in safe_goal_inds
        ]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_ind):
        path = self.end.state
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path = torch.cat([path, node.state], dim=0)
            node = node.parent
        path = torch.cat([path, node.state], dim=0)

        return path

    def cal_state_dist(self, from_state, to_state):
        delta_state = to_state - from_state
        delta_state[:,self.rotation_index] = angle_normalize(delta_state[:,self.rotation_index])
        return torch.norm(delta_state)

    def cal_state_dist_batch(self, state_list, init_state):
        delta_state = state_list - init_state
        delta_state[:,self.rotation_index] = angle_normalize(delta_state[:,self.rotation_index])

        return torch.norm(delta_state, dim=1)

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

        if rnd is not None:
            plt.plot(rnd.state[0, 0], rnd.state[0, 1], "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path[:, 0], node.path[:, 1], "-g")

        # for state, size in obstacle_list:
        #     self.plot_circle(state[0], state[1], size)

        plt.plot(self.start.state[0, 0], self.start.state[0, 1], "xr")
        plt.plot(self.end.state[0, 0], self.end.state[0, 1], "xr")
        plt.axis("equal")
        plt.axis(
            [
                self.min_rand[0].item(),
                self.max_rand[0].item(),
                self.min_rand[1].item(),
                self.max_rand[1].item(),
            ]
        )
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)


if __name__ == "__main__":

    obstacle_list = [
        (torch.tensor([5.0, 5]), 1),
        (torch.tensor([3.0, 6]), 2),
        # (torch.tensor([3.0, 8]), 2),
        # (torch.tensor([3.0, 10]), 2),
        # (torch.tensor([7.0, 5]), 2),
        # (torch.tensor([9.0, 5]), 2),
        # (torch.tensor([8.0, 10]), 1),
        # (torch.tensor([6.0, 12]), 1),
    ]
    
    def collision_check(state):
        for pos, size in obstacle_list:
            distances = torch.norm(state - pos.view(1, -1))
    
            if distances <= size + 0.1:
                return False  # collision
    
        return True  # safe
    
    rrt_star = RRTStar(
        start=torch.tensor([0.0, 0]).unsqueeze(0),
        goal=torch.tensor([6.0, 10]).unsqueeze(0),
        dim=2,
        max_state=torch.tensor([7,10]),
        min_state=torch.tensor([-1,-1]),
        expand_dis=1,
        max_iter=1000,
        collision_check=collision_check,
        connect_circle_dist=1.0,
        rotation_index = [],
    )
    path = rrt_star.planning(debug=False)
    
    import time
    start_time = time.time()
    print(path)
    print(f"time cost: {time.time() - start_time}")
    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
    
        rrt_star.draw_graph()
        plt.plot(path[:, 0], path[:, 1], "r--")
        plt.grid(True)
        plt.show()

    # import numpy as np
    # import roboticstoolbox as rtb
    # import spatialgeometry as sg
    # import swift

    # from content.assets.collision_swift.bookshelf import collisions

    # start = np.asarray([0, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    # goal = np.asarray([5, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

    # dt = 0.025
    # arrived = False
    # safe_dist = 0.01

    # robot = rtb.models.FrankieOmni()
    # start_link = robot.links[0]
    # end_link = robot.link_dict["panda_link8"]

    # robot.q = start
    # q_min = torch.from_numpy(robot.qlim[0, :])
    # q_max = torch.from_numpy(robot.qlim[1, :])
    # q_min[0] = -10
    # q_max[0] = 10
    # q_min[1] = -10
    # q_max[1] = 10
    # q_min[2] = -np.pi
    # q_max[2] = np.pi

    # goal_T = robot.fkine(goal)
    # ax_goal = sg.Axes(0.1)
    # ax_goal.T = goal_T

    # def collision_check(state):
    #     """

    #     Args:
    #         state: torch.tensor

    #     Returns:

    #     """
    #     robot._update_link_tf(state[: robot.n].numpy())
    #     robot._propogate_scene_tree()
    #     for collision in collisions:
    #         end, start, _ = robot._get_limit_links(start=start_link, end=end_link)

    #         links, n, _ = robot.get_path(start=start, end=end)

    #         j = 0

    #         for link in links:
    #             if link.isjoint:
    #                 j += 1

    #             col_list = link.collision

    #             for link_col in col_list:
    #                 d, wTlp, wTcp = link_col.closest_point(collision, safe_dist)

    #                 if d is not None and wTlp is not None and wTcp is not None:
    #                     return False

    #     return True

    # planner = RRTStar(
    #     start=torch.tensor(start).unsqueeze(0),
    #     goal=torch.tensor(goal).unsqueeze(0),
    #     dim=robot.n,
    #     max_state=q_max,
    #     min_state=q_min,
    #     collision_check=collision_check,
    #     expand_dis=0.5,
    #     path_resolution=0.1,
    #     max_iter=1000,
    #     connect_circle_dist=1.0,
    # )

    # env = swift.Swift()
    # env.launch(realtime=True)
    # env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])

    # env.add(ax_goal)

    # for col in collisions:
    #     env.add(col)
    # env.add(robot, robot_alpha=1.0, collision_alpha=0.0)

    # env.step()

    # path = planner.planning()
    # path = torch.flip(path, dims=[0])
    # print("path:", path)

    # for i in range(path.size(0)):
    #     robot.q = path[i, :].numpy()
    #     env.step()

    # env.hold()
