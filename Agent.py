import numpy as np
import pybullet as p
import time
import pandas as pd
class Agent:
    def __init__(self, env, target_pos, eps=0.01, control_method="end"):
        self.env = env
        self.robot = env.ur5
        self.camera = env.camera

        self.target_pos = np.array(target_pos)
        self.eps = eps
        self.done = False
        self.photo_taken = False
        self.control_method = control_method

        # 🔴 RUCH ROBOTA RAZ NA START
        # self.robot.move_arm(self.target_pos, control_method=self.control_method)

    def step(self):
        if self.done:
            return

        # if self.control_method == "joint":g
        #     # sprawdzenie odległości w przestrzeni stawów
        #     current_joints = np.array([p.getJointState(self.robot.robot_id, j)[0]
        #                                for j in self.robot.arm_controllable_joints])
        #     dist = np.linalg.norm(current_joints - self.target_pos)
        # else:

        self.env.ur5.move_arm([-1.54, -1.54, 1.54, -1.54, -1.54, -1.54], 'joint')

        # sprawdzenie odległości TCP
        link_state = p.getLinkState(
            self.robot.robot_id,
            self.robot.eef_id,
            computeForwardKinematics=True
        )
        tcp_pos = np.array(link_state[4])
        dist = np.linalg.norm(tcp_pos - self.target_pos)

        print(tcp_pos, self.target_pos)
        if dist < self.eps:
            print("✅ Cel osiągnięty")

            if not self.photo_taken:
                time.sleep(1)

                rgbd = self.camera.shot_rgbd(self.robot)
                print("📸 Zdjęcie wykonane:", rgbd.shape)
                self.photo_taken = True
                self.done = True