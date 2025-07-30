"""
Composite inverse-kinematics solver for an AGV base + JAKA ZU7 arm.

Configuration vector:
    q = [x, y, θ, q0, q1, q2, q3, q4, q5]
        └─ AGV ───┘  └────── manipulator ───────┘
"""

import os
import sys
import copy
import numpy as np
from math import pi

# ------------------------------------------------------------------
#  Local package imports (adjust the relative path to your project)
# ------------------------------------------------------------------
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, '../motion_planning'))

from agv_kinematics import agv_arm            # AGV + arm forward kinematics
from jaka_IK import (                         # JAKA ZU-7 DH utilities
    get_DH_matrix_JAKA_ZU7,
    get_DH_matrix_JAKA_ZU7_ur,
    forward_kinematic_DH_solution,
    JAKA_inverse_kinematic_DH_solution,
)

# ------------------------------------------------------------------
#  Manipulator-only IK wrapper
# ------------------------------------------------------------------
class JakaZU7IK:
    """Closed-form IK for the 6-DoF JAKA ZU-7 arm."""

    def __init__(self) -> None:
        self.DH    = get_DH_matrix_JAKA_ZU7()
        self.DH_ur = get_DH_matrix_JAKA_ZU7_ur()
        self.lower = np.array([-pi, -pi, -pi, -pi, -2 * pi, -2 * pi])
        self.upper = -self.lower

    # ---------- forward kinematics ----------
    def fk(self, q: np.ndarray) -> np.ndarray:
        """4×4 pose of the wrist TCP given joint vector q (len=6)."""
        return forward_kinematic_DH_solution(self.DH, q)

    # ---------- inverse kinematics ----------
    def ik(self, T06: np.ndarray) -> np.ndarray:
        """Return all IK solutions within joint limits (shape: N×6)."""
        sols = []
        for q in JAKA_inverse_kinematic_DH_solution(self.DH, self.DH_ur, T06).T:
            q = np.clip(q, self.lower, self.upper)           # enforce limits
            if np.linalg.norm(self.fk(q) - T06) < 1e-6:      # verify
                sols.append(q)
        return np.asarray(sols)


# ------------------------------------------------------------------
#  Composite AGV + arm IK
# ------------------------------------------------------------------
class AGVIK:
    """
    Grid-search IK for a mobile base (SE(2)) plus JAKA ZU-7 arm.

    Parameters
    ----------
    workspace : float
        Half-width of the square search window (m) around the target.
    grid : int
        Number of interpolation points per dimension (x, y, θ).
    residual_tol : float
        Pose error tolerance for accepting a solution (Frobenius norm).
    """

    def __init__(self) -> None:
        self.agv_arm = agv_arm()           # FK for the composite system
        self.jik = JakaZU7IK()         # manipulator IK
        self.T_agv2arm = self.agv_arm.T_agv_2_arm   # static transform

    # ---------- main solver ----------
    def solve(
        self,
        target_T: np.ndarray,
        workspace: float = 0.6,
        grid: int = 10,
        residual_tol: float = 1e-2,
        return_all_flag: bool = False
    ):
        """Return best configuration q (len=9) or None if not found."""
        cands = []

        # 1 Generate SE(2) samples for the AGV base
        x_samples   = np.linspace(-workspace, workspace, grid) + target_T[0, 3]
        y_samples   = np.linspace(-workspace, workspace, grid) + target_T[1, 3]
        yaw_samples = np.linspace(-pi,        pi,        grid)

        # 2 Enumerate every base pose and run arm IK
        for x in x_samples:
            for y in y_samples:
                for yaw in yaw_samples:
                    now_q = [0 for i in range(9)]
                    now_q[0:2] = [x,y]
                    now_q[2] = yaw
                    T_chain = self.agv_arm.FK(now_q)[2]   # pose after 2nd frame
                    T_goal  = np.linalg.inv(T_chain @ self.T_agv2arm) @ target_T

                    for q_arm in self.jik.ik(T_goal):
                        cands.append(np.concatenate([now_q[0:3], q_arm]))

        if not cands:
            return None

        if return_all_flag:
            return cands
        # 3 Score candidates: residual first (filter), then joint effort
        res    = np.array([np.linalg.norm(self.agv_arm.FK(q)[-1] @ np.linalg.inv(target_T) - np.eye(4)) for q in cands])
        effort = np.array([np.sum(np.abs(q[3:])) for q in cands])

        feas = np.where(res < residual_tol)[0]
        if feas.size == 0:
            return None

        best = cands[feas[np.argmin(effort[feas])]]
        # T_fk = self.agv_arm.FK(best)
        # self.agv_arm.vis_collision()


        return best


# ------------------------------------------------------------------
#  Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    solver = AGVIK()

    # 1 Generate a reachable target pose from an arbitrary seed q
    seed_q = np.zeros(9)
    target_pose = solver.agv_arm.FK(seed_q)[-1]

    # 2 Solve IK
    best_q = solver.solve(target_pose, workspace=0.6, grid=10, residual_tol=1e-2)

    if best_q is None:
        print("No feasible solution found.")
    else:
        print("Best joint configuration:\n", best_q)
        # Optional – visualize for collision checking
        T_fk = solver.agv_arm.FK(best_q)
        solver.agv_arm.vis_collision()
