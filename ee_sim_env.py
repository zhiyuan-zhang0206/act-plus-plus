import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose, sample_stir_pose, sample_openlid_pose, sample_transfercube_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed
def task_name_to_task_class(task_name):
    if task_name == 'stir':
        return StirEETask
    elif task_name == 'openlid':
        return OpenLidEETask
    elif task_name == 'transfercube':
        return TransferCubeEETask
    else:
        raise NotImplementedError

def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    #
    xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_{task_name}.xml')
    if not os.path.exists(xml_path):
        raise NotImplementedError(f"{xml_path} does not exist")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = task_name_to_task_class(task_name)(random=False)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)

    return env

from dm_control.utils import inverse_kinematics as ik

    # qpos_left = qpos_from_site_pose(physics,
    #                                 site_name=left_site_name,
    #                                 target_pos=target_pos_left,
    #                                 target_quat=target_quat_left,
    #                                 joint_names=joint_names_left,
    #                                 max_steps=max_steps)

    # # Compute qpos for the right end effector
    # qpos_right = qpos_from_site_pose(physics,
    #                                  site_name=right_site_name,
    #                                  target_pos=target_pos_right,
    #                                  target_quat=target_quat_right,
    #                                  joint_names=joint_names_right,
    #                                  max_steps=max_steps)

from pyquaternion import Quaternion
def rotate_z_pi(quat):
    q = Quaternion(quat)
    axis = q.axis

    # Create the symmetric axis
    symmetric_axis = np.array([-axis[0], axis[1], axis[2]])

    # Create the symmetric quaternion
    symmetric_q = Quaternion(axis=symmetric_axis, angle=q.angle)
    return symmetric_q.elements

def get_qpos_ik(physics, action_left, action_right):
    qpos_left = ik.qpos_from_site_pose(physics, site_name='vx300s_left/gripper_link_site', 
                               target_pos= action_left[:3],
                               target_quat= action_left[3:7], 
                               joint_names=[
                                        "vx300s_left/waist",
                                        "vx300s_left/shoulder",
                                        "vx300s_left/elbow",
                                        "vx300s_left/forearm_roll",
                                        "vx300s_left/wrist_angle",
                                        "vx300s_left/wrist_rotate"
                                    ],
                               max_steps=1000)
    qpos_right = ik.qpos_from_site_pose(physics, site_name='vx300s_right/gripper_link_site', 
                               target_pos= action_right[:3],
                            #    target_quat= rotate_z_pi(action_right[3:7]), 
                               target_quat= action_right[3:7], 
                               joint_names=[
                                        "vx300s_right/waist",
                                        "vx300s_right/shoulder",
                                        "vx300s_right/elbow",
                                        "vx300s_right/forearm_roll",
                                        "vx300s_right/wrist_angle",
                                        "vx300s_right/wrist_rotate"
                                    ],
                               max_steps=1000)
    return qpos_left.qpos[0:6], qpos_right.qpos[8:14]

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        qpos_left, qpos_right = get_qpos_ik(physics, action_left, action_right)
        np.copyto(physics.data.qpos[0:6], qpos_left)
        np.copyto(physics.data.qpos[8:14], qpos_right)
        # np.copyto(physics.data.mocap_pos[0], action_left[:3])
        # np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # # right
        # np.copyto(physics.data.mocap_pos[1], action_right[:3])
        # np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics, close_width=None):
        # reset joint position
        # physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881+0.1, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881-0.1, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])
        
        action_left = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        action_right = np.concatenate([physics.data.mocap_pos[1], np.array([0, 0 , 0, -1])]).copy()
        q_left, q_right = get_qpos_ik(physics, action_left, action_right)
        # np.copyto(physics.data.qpos[0:6], q_left)
        # np.copyto(physics.data.qpos[8:14], q_right)
        physics.named.data.qpos[:16] = START_ARM_POSE
        physics.named.data.qpos[0:6] = q_left
        physics.named.data.qpos[8:14] = q_right
        # from loguru import logger
        # logger.critical(f"qpos: {physics.data.qpos}")
        # import sys
        # sys.exit()
        # reset gripper control
        if close_width is not None:
            close_gripper_control = np.array([
                close_width,
                -close_width,
                close_width,
                -close_width,
            ])
        else:
            close_gripper_control = np.array([
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
                PUPPET_GRIPPER_POSITION_CLOSE,
                -PUPPET_GRIPPER_POSITION_CLOSE,
            ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        # obs['images']['horizontal'] = physics.render(height=300, width=300, camera_id='horizontal')
        # obs['images']['angle'] = physics.render(height=300, width=300, camera_id='angle')
        # obs['images']['front_close'] = physics.render(height=300, width=300, camera_id='front_close')
        # obs['images']['left_angle'] = physics.render(height=300, width=300, camera_id='left_angle')
        # obs['images']['right_angle'] = physics.render(height=300, width=300, camera_id='right_angle')
        # obs['images']['left_wrist'] = physics.render(height=300, width=300, camera_id='left_wrist')
        # obs['images']['right_wrist'] = physics.render(height=300, width=300, camera_id='right_wrist')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        obs['left_pose'] = np.concatenate([physics.named.data.site_xpos["vx300s_left/gripper_link_site"].copy(), physics.named.data.xquat['vx300s_left/gripper_link'].copy()])
        obs['right_pose'] = np.concatenate([physics.named.data.site_xpos["vx300s_right/gripper_link_site"].copy(), physics.named.data.xquat['vx300s_right/gripper_link'].copy()])
        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2
        self.objects_start_pose = sample_transfercube_pose()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics, close_width=0.044)
        
        cube_pose = self.objects_start_pose[0:7]
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        cube_start_id = physics.model.name2id('cube_joint', 'joint')
        cube_start_idx = id2index(cube_start_id)
        np.copyto(physics.data.qpos[cube_start_idx : cube_start_idx + 7], cube_pose)
        dummy_pose = self.objects_start_pose[7:14]
        dummy_start_id = physics.model.name2id('dummy_joint', 'joint')
        dummy_start_idx = id2index(dummy_start_id)
        np.copyto(physics.data.qpos[dummy_start_idx : dummy_start_idx + 7], dummy_pose)
        object_info = {'object_num':2, 
                       'object_poses':np.concatenate([cube_pose, dummy_pose])}
        self.object_info = object_info
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky
        cube_start_id = physics.model.name2id('cube_joint', 'joint')
        cube_start_idx = id2index(cube_start_id)

        cube_pose = physics.data.qpos[cube_start_idx : cube_start_idx + 7].copy()

        env_state = np.concatenate([cube_pose])
        return env_state

    def get_reward(self, physics):
        return 4


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

class StirEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.objects_start_pose = sample_stir_pose()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics, close_width=0.044)
        
        cup_pose = self.objects_start_pose[0:7]
        spoon_pose = self.objects_start_pose[7:14]
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        cup_start_id = physics.model.name2id('cup_joint', 'joint')
        cup_start_idx = id2index(cup_start_id)
        spoon_start_id = physics.model.name2id('spoon_joint', 'joint')
        spoon_start_idx = id2index(spoon_start_id)
        np.copyto(physics.data.qpos[cup_start_idx : cup_start_idx + 7], cup_pose)

        np.copyto(physics.data.qpos[spoon_start_idx : spoon_start_idx + 7], spoon_pose)
        # print(f"randomized cube position to {cube_position}")
        object_info = {'object_num':2, 
                       'object_poses':np.concatenate([cup_pose, spoon_pose])}
        self.object_info = object_info
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky
        cup_start_id = physics.model.name2id('cup_joint', 'joint')
        cup_start_idx = id2index(cup_start_id)
        spoon_start_id = physics.model.name2id('spoon_joint', 'joint')
        spoon_start_idx = id2index(spoon_start_id)
        cup_pose = physics.data.qpos[cup_start_idx : cup_start_idx + 7].copy()
        spoon_pose = physics.data.qpos[spoon_start_idx : spoon_start_idx + 7].copy()
        env_state = np.concatenate([cup_pose, spoon_pose])
        return env_state

    def get_reward(self, physics):
        return 4
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        
        return reward

class OpenLidEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.objects_start_pose = sample_openlid_pose()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics, close_width=0.044)
        
        cup_pose = self.objects_start_pose[0:7]
        lid = self.objects_start_pose[7:14]
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        cup_start_id = physics.model.name2id('cuplid_cup_joint', 'joint')
        cup_start_idx = id2index(cup_start_id)
        lid_start_id = physics.model.name2id('cuplid_lid_joint', 'joint')
        lid_start_idx = id2index(lid_start_id)
        np.copyto(physics.data.qpos[cup_start_idx : cup_start_idx + 7], cup_pose)

        np.copyto(physics.data.qpos[lid_start_idx : lid_start_idx + 7], lid)
        # print(f"randomized cube position to {cube_position}")
        object_info = {'object_num':2, 
                       'object_poses':np.concatenate([cup_pose, lid])}
        self.object_info = object_info
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky
        cup_start_id = physics.model.name2id('cuplid_cup_joint', 'joint')
        cup_start_idx = id2index(cup_start_id)
        spoon_start_id = physics.model.name2id('cuplid_lid_joint', 'joint')
        spoon_start_idx = id2index(spoon_start_id)
        cup_pose = physics.data.qpos[cup_start_idx : cup_start_idx + 7].copy()
        spoon_pose = physics.data.qpos[spoon_start_idx : spoon_start_idx + 7].copy()
        env_state = np.concatenate([cup_pose, spoon_pose])
        return env_state

    def get_reward(self, physics):
        return 4
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        
        return reward
