import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name, object_info:dict=None)->control.Environment:
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (300x300x3)}        # h, w, c, dtype='uint8'
    """
    # if 'sim_transfer_cube' in task_name:
    #     xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
    #     physics = mujoco.Physics.from_xml_path(xml_path)
    #     task = TransferCubeTask(random=False)
    #     env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
    #                               n_sub_steps=None, flat_observation=False)
    # elif 'sim_insertion' in task_name:
    #     xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
    #     physics = mujoco.Physics.from_xml_path(xml_path)
    #     task = InsertionTask(random=False)
    #     env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
    #                               n_sub_steps=None, flat_observation=False)
    if task_name == 'stir':
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_stir.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = StirTask(random=False, object_info=object_info)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif task_name == 'openlid':
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_openlid.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = OpenLidTask(random=False, object_info=object_info)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif task_name == 'transfercube':
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfercube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False, object_info=object_info)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.render = True
        self.last_left_image = np.zeros((300,300,3), dtype=np.uint8)
        self.last_right_image = np.zeros((300,300,3), dtype=np.uint8)

    def set_render_state(self, render:bool):
        self.render = render

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        # obs['images']['horizontal'] = physics.render(height=300, width=300, camera_id='horizontal')
        # obs['images']['left_wrist'] = physics.render(height=300, width=300, camera_id='left_wrist')
        # obs['images']['right_wrist'] = physics.render(height=300, width=300, camera_id='right_wrist')
        # obs['images']['angle'] = physics.render(height=300, width=300, camera_id='angle')
        # obs['images']['front_close'] = physics.render(height=300, width=300, camera_id='front_close')
        if self.render:
            obs['images']['left_angle'] = physics.render(height=300, width=300, camera_id='left_angle')
            obs['images']['right_angle'] = physics.render(height=300, width=300, camera_id='right_angle')
            self.last_left_image = obs['images']['left_angle'].copy()
            self.last_right_image = obs['images']['right_angle'].copy()
        else:
            obs['images']['left_angle'] = self.last_left_image
            obs['images']['right_angle'] = self.last_right_image
        # <geom condim="0" contype="0" pos="0 0 0" size="0.02 0.02 0.02" type="box" name="left_dummy" rgba="1 0 0 0" />
        obs['left_pose'] = np.concatenate([physics.named.data.site_xpos["vx300s_left/gripper_link_site"].copy(), physics.named.data.xquat['vx300s_left/gripper_link'].copy()])
        obs['right_pose'] = np.concatenate([physics.named.data.site_xpos["vx300s_right/gripper_link_site"].copy(), physics.named.data.xquat['vx300s_right/gripper_link'].copy()])
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError



class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
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

class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None, object_info:dict=None):
        super().__init__(random=random)
        self.max_reward = 2
        self.object_info = object_info

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            physics.named.data.qpos[-7*self.object_info['object_num']:] = self.object_info['object_poses']
            # assert BOX_POSE[0] is not None
            # physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return 4
        all_contact_pairs = get_contact_pairs(physics)

        touch_right_gripper = tuple(sorted(["cube", "vx300s_right/10_right_gripper_finger"])) in all_contact_pairs
        touch_left_gripper = tuple(sorted(["cube", "vx300s_left/10_left_gripper_finger"])) in all_contact_pairs     
                             
        reward = int(not touch_left_gripper) + int(touch_right_gripper)

        return reward


# from ee_sim_env import get_qpos_ik
class StirTask(BimanualViperXTask):
    def __init__(self, random=None, object_info:dict=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.object_info = object_info

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            physics.named.data.qpos[-7*self.object_info['object_num']:] = self.object_info['object_poses']
            # assert BOX_POSE[0] is not None
            # physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = get_contact_pairs(physics)

        touch_right_gripper = tuple(sorted(["spoon_handle_collision", "vx300s_right/10_right_gripper_finger"])) in all_contact_pairs
        touch_left_gripper = tuple(sorted(["cup_handle_collision", "vx300s_left/10_left_gripper_finger"])) in all_contact_pairs     
                             

        closeness_reward = 0
        object_pose = physics.data.qpos.copy()[16:]
        cup_location = object_pose[:3]
        spoon_location = object_pose[7:10]
        x_y_distance = np.linalg.norm(cup_location[:2] - spoon_location[:2])
        z_distance = np.abs(cup_location[2] - spoon_location[2])
        weighted_distance = (x_y_distance * 2 + z_distance) / 2
        closeness_reward = (0.2 - np.clip(weighted_distance, 0, 0.2)) * 10
        reward = 0
        reward += int(touch_left_gripper)
        reward += int(touch_right_gripper)
        reward += closeness_reward

        return reward

def get_contact_pairs(physics):
    all_contact_pairs = set()
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
        name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
        contact_pair = tuple(sorted([name_geom_1, name_geom_2]))
        all_contact_pairs.add(contact_pair)
    return all_contact_pairs

class OpenLidTask(BimanualViperXTask):
    def __init__(self, random=None, object_info:dict=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.object_info = object_info

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            physics.named.data.qpos[-7*self.object_info['object_num']:] = self.object_info['object_poses']
            # assert BOX_POSE[0] is not None
            # physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = get_contact_pairs(physics)
        reward = 0
        lid_collision_names = [
            "cuplid_lid_0_collision",
            "cuplid_lid_1_collision",
            "cuplid_lid_2_collision",
            "cuplid_lid_3_collision",
            "cuplid_lid_4_collision",
            "cuplid_lid_5_collision",
            "cuplid_lid_6_collision",
            "cuplid_lid_7_collision",
            "cuplid_lid_cylinder_collision",
        ]
        touch_right_gripper = any([tuple(sorted([lid_name, "vx300s_right/10_right_gripper_finger"])) in all_contact_pairs for lid_name in lid_collision_names])
        touch_right_gripper |= any([tuple(sorted([lid_name, "vx300s_right/10_left_gripper_finger"])) in all_contact_pairs for lid_name in lid_collision_names])
        # print(touch_right_gripper)
        cup_collision_names = [
            "cuplid_cup_collision",
            "cuplid_cup_collision_1",
            "cuplid_cup_collision_2",
        ]
        touch_left_gripper = any([tuple(sorted([cup_name, "vx300s_left/10_left_gripper_finger"])) in all_contact_pairs for cup_name in cup_collision_names])
        touch_right_gripper |= any([tuple(sorted([cup_name, "vx300s_left/10_right_gripper_finger"])) in all_contact_pairs for cup_name in cup_collision_names])
        reward += int(touch_left_gripper)
        reward += int(touch_right_gripper)
        object_pose = physics.data.qpos.copy()[16:]
        cup_location = object_pose[:3]
        lid_location = object_pose[7:10]
        x_y_distance = np.linalg.norm(cup_location[:2] - lid_location[:2])
        farness_reward = np.clip(x_y_distance, 0, 0.1) * 20
        reward += farness_reward
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

