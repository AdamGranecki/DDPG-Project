import pybullet_data
import time
import math
from collections import namedtuple
import numpy as np
import pybullet as p
from Agent import Agent

class UR5Robot:
    def __init__(self, table_id, table_height):
        self.table_id = table_id
        self.table_height = table_height

        # ---- LOAD ROBOT ----
        self.base_position = [0, 0, table_height + 0.1]
        self.base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.start_poses = [0, 0, -1.54, 1.54, -1.54, -1.54]
        self.arm_rest_poses = list(self.start_poses)
        self.gripper_range = [0, 0.085]

        self.robot_id = p.loadURDF(
            "ur5_robotiq_85.urdf",
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=False
        )

        self._attach_to_table()

        # ---- ARM JOINTS ----
        self.arm_joint_indices = [1, 2, 3, 4, 5, 6]

        self.__parse_joint_info__()
        self.arm_target_positions = list(self.start_poses)
        self.arm_max_forces = {
            j: self.joints[j].maxForce for j in self.arm_joint_indices
        }
        self.move_arm(self.start_poses, 'joint')

        # ---- GRIPPER ----
        self.gripper_target_angle = None
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
        self.open_gripper()

    def _attach_to_table(self):
        # Przymocowanie robota do stołu (constraint)
        p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.robot_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, self.table_height / 2],
            childFramePosition=[0, 0, 0]
        )

    # Ustaw robota na pozycję startową
    def move_arm(self, action, control_method):
        if control_method == 'end':
            x, y, z = action
            roll, pitch, yaw = [0, math.pi / 2, 0]
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits,
                                                       self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action

        self.arm_target_positions = list(joint_poses)

        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.robot_id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)


    def hold_arm(self):
        for j, q in zip(self.arm_joint_indices, self.arm_target_positions):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=self.arm_max_forces[j]
            )


    def get_id(self):
        return self.robot_id

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.robot_id)
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.robot_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.robot_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        print("self.arm_controllable_joints:  ",self.arm_controllable_joints)
        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][
                                :self.arm_num_dofs]


    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if
                                       joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.robot_id, self.mimic_parent_id,
                                   self.robot_id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100,
                               erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def move_gripper(self, open_length):
        open_length = np.clip(open_length, *self.gripper_range)

        self.gripper_target_angle = (
                0.715 - math.asin((open_length - 0.010) / 0.1143)
        )

        joint = self.joints[self.mimic_parent_id]

        p.setJointMotorControl2(
            self.robot_id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=self.gripper_target_angle,
            force=joint.maxForce,
            maxVelocity=joint.maxVelocity
        )

    def hold_gripper(self):
        if self.gripper_target_angle is None:
            return

        joint = self.joints[self.mimic_parent_id]

        p.setJointMotorControl2(
            self.robot_id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=self.gripper_target_angle,
            force=joint.maxForce,
            maxVelocity=joint.maxVelocity
        )

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])


class Environment:
    def __init__(self, gui=True, hz=240):
        self.gui = gui
        self.hz = hz
        self.dt = 1.0 / hz

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)

        self._load_world()
        self._load_robot()

    def _load_world(self):
        self.plane_id = p.loadURDF("plane.urdf")

        self.table_height = 0.5
        self.table_position = [0, 0, 0]
        self.table_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=self.table_position,
            baseOrientation=self.table_orientation,
            useFixedBase=True
        )

    def _load_robot(self):
        self.ur5 = UR5Robot(self.table_id, self.table_height)
        self.camera = Camera(
            cam_pos=[1.2, 0, 1.2],
            cam_tar=[0, 0, 0.5],
            cam_up_vector=[0, 0, 1],
            near=0.01,
            far=3.0,
            size=(640, 480),
            fov=60
        )
    def step(self):
        self.ur5.hold_arm()
        self.ur5.hold_gripper()

        p.stepSimulation()
        # if self.gui:
        #     time.sleep(self.dt)

    def close(self):
        p.disconnect(self.client_id)

class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def configure_view_from_robot(self, robot):
        ls = p.getLinkState(robot.robot_id, robot.eef_id, computeForwardKinematics=1)
        cam_pos, cam_orn = ls[4], ls[5]

        offset_local = [.1, 0, 0]
        offset_world = p.rotateVector(cam_orn, offset_local)
        cam_pos = [cam_pos[i] + offset_world[i] for i in range(3)]

        tilt_angle = math.radians(5)
        tilt_quat = p.getQuaternionFromAxisAngle([1, 0, 0], tilt_angle)
        _, cam_orn = p.multiplyTransforms([0, 0, 0], tilt_quat, [0, 0, 0], cam_orn)

        rot = p.getMatrixFromQuaternion(cam_orn)
        forward = [rot[0], rot[3], rot[6]]
        up = [rot[2], rot[5], rot[8]]

        # # draw the camera coordinate
        # p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[i] for i in range(3)], [1, 0, 0], 2)  # X - czerwony
        # p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[3 + i] for i in range(3)], [0, 1, 0], 2)  # Y - zielony
        # p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[6 + i] for i in range(3)], [0, 0, 1], 2)  # Z - niebieski

        self.view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
                                               cameraTargetPosition=[cam_pos[i] + forward[i] for i in range(3)],
                                               cameraUpVector=up)

        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                              aspect=self.width / self.height,
                                                              nearVal=self.near,
                                                              farVal=self.far)

    def shot_rgbd(self, robot):
        self.configure_view_from_robot(robot)
        w, h = self.width, self.height
        img = p.getCameraImage(w, h, self.view_matrix, self.projection_matrix,
                               renderer=p.ER_TINY_RENDERER)
        rgb, depth, seg = img[2], img[3], img[4]

        # depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        depth_expanded = depth[..., None]
        # depth_expanded = depth_expanded.astype(np.uint8)[..., None]

        rgbd = np.concatenate([rgb[:, :, :3]/255, depth_expanded], axis=2)
        return rgbd



# Pętla symulacji
env = Environment(gui=True)

agent = Agent(
    env=env,
    target_pos=[.3,.8,1],  # pozycja B
    eps=0.1
)

while True:
    env.step()      # symulacja + hold
    agent.step()    # logika ruchu + zdjęcie

    if agent.done:
        print("🛑 Zatrzymanie programu")
        break

time.sleep(10)
env.close()

