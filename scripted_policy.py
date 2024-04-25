import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from loguru import logger
from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
import random
# import IPython
# e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_generated = False

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0 and not self.trajectory_generated:
            self.generate_trajectory(ts)
        if not self.trajectory_generated:
            raise ValueError("Trajectory not generated yet")

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        # print(f"{self.step_count=}, {action_left=}, {action_right=}")
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]

class StirPolicy(BasePolicy):
    language_instruction :str = 'put spoon into cup' # 'use spoon to stir coffee'
    def generate_trajectory(self, ts_first, random_values:list=None):
        self.trajectory_generated = True
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        left_initial_loc = init_mocap_pose_left[:3]
        right_initial_loc = init_mocap_pose_right[:3]
        cup_info = np.array(ts_first.observation['env_state'])[:7]
        cup_xyz = cup_info[:3]
        cup_quat = cup_info[3:]

        spoon_info = np.array(ts_first.observation['env_state'])[7:]
        spoon_xyz = spoon_info[:3].copy()
        spoon_xyz[2] = 0
        spoon_quat = spoon_info[3:]


        # gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        # # gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=0)
        # # gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        # # gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        # gripper_stir_quat_right =  Quaternion(init_mocap_pose_right[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # meet_xyz = np.array([0.2, 0.7, 0.3])
        # lift_right = 0.00715

        left_initial_quat = Quaternion(np.array([1, 0 , 0, 0]))
        hold_deltas = [
            np.array([-0.0, 0, -0.1]),
            np.array([-0.0, 0, -0.1]),
            np.array([0.01, 0, -0.1]),
            np.array([+0.02, 0, -0.1]),
        ]
        stir_deltas = [
            np.array([-0.02, 0.01,     0.03]),
            np.array([-0.01, -0.01,    0.03]),
            np.array([0.01, 0.01,    0.03]),
            np.array([-0.01, -0.01,    0.04]),
            np.array([-0.02, 0.01,    0.04]),
            np.array([-0.02, 0.01,    0.05]),
            np.array([-0.01, -0.01,    0.06]),
        ]
        if random_values is None:
            meet_xyz = (cup_xyz + spoon_xyz)/2 + np.random.uniform(-0.1, 0.1, 3)
            meet_xyz[2] = 0.3 + np.random.uniform(-0.1, 0.1)
            delta_1 = np.random.uniform(low=-0.2, high=0., size=3)
            delta_6 = np.random.uniform(low=0, high=0.2, size=3)
            random_values = {
                "meet_xyz": meet_xyz,
                "delta_1": delta_1,
                "delta_2": random.choice(hold_deltas),
                "delta_3": random.choice(hold_deltas),
                "delta_4": random.choice(hold_deltas),
                "delta_5": random.choice(hold_deltas),
                "delta_6": delta_6,
                "delta_7":random.choice(stir_deltas),
                "delta_8":random.choice(stir_deltas),
                "delta_9":random.choice(stir_deltas),
                "delta_10":random.choice(stir_deltas),
            }
        else:
            logger.info(f"Using given random values.")
        # separate_xyz = meet_xyz + np.array([0.1, 0, 0.1])
        
        self.left_trajectory = [
            {"t": 0,    "xyz": left_initial_loc,                            "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 50,  "xyz": cup_xyz + np.array([-0.01, 0.0, 0.2]),        "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 100,  "xyz": cup_xyz + np.array([-0.01, 0.0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 150,  "xyz": cup_xyz +   np.array([-0.01, 0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 0}, 
            # {"t": 200,  "xyz": cup_xyz +   np.array([-0.01, 0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 200,  "xyz": random_values['meet_xyz'] +  random_values['delta_1'],    "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 300,  "xyz": random_values['meet_xyz'] +  random_values['delta_2'] ,      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 330,  "xyz": random_values['meet_xyz'] +  random_values['delta_3'] ,      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 360,  "xyz": random_values['meet_xyz'] +  random_values['delta_4'] ,      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 400,  "xyz": random_values['meet_xyz'] +  random_values['delta_5'] ,    "quat": left_initial_quat.elements,            "gripper": 0}, 
        ]

        right_initial_quat = Quaternion(np.array([0, 0 , 0, -1]))
        right_down = right_initial_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)
        right_stir = right_initial_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)

        self.right_trajectory = [
            {"t": 0,   "xyz": right_initial_loc,                            "quat": right_initial_quat.elements,          "gripper": 1}, # sleep
            {"t": 50, "xyz": spoon_xyz + np.array([ -0.08, 0.0, -0.032]),    "quat": right_down.elements,   "gripper": 1}, # sleep
            {"t": 100, "xyz": spoon_xyz + np.array([-0.08, 0.0, -0.032]),    "quat": right_down.elements,   "gripper": 0}, # sleep
            {"t": 150, "xyz": spoon_xyz + np.array([-0.08, 0.0, 0.2]),      "quat": right_down.elements,   "gripper": 0}, # sleep
            {"t": 200, "xyz": spoon_xyz + np.array([-0.08, 0.0, 0.2]),      "quat": right_stir.elements,   "gripper": 0}, # sleep
            {"t": 250, "xyz": random_values['meet_xyz'] + random_values['delta_6'],                           "quat": right_stir.elements,   "gripper":0}, # sleep
            {"t": 300, "xyz": random_values['meet_xyz'] + np.array([-0.01, 0, 0.13]),          "quat": right_stir.elements,   "gripper":0}, # sleep
            {"t": 350, "xyz": random_values['meet_xyz'] + random_values['delta_7'],          "quat": right_stir.elements,   "gripper":0}, # sleep
            {"t": 370, "xyz": random_values['meet_xyz'] + random_values['delta_8'],          "quat": right_stir.elements,   "gripper":0}, # sleep
            {"t": 390, "xyz": random_values['meet_xyz'] + random_values['delta_9'],          "quat": right_stir.elements,   "gripper":0}, # sleep
            {"t": 400, "xyz": random_values['meet_xyz'] + random_values['delta_10'],          "quat": right_stir.elements,   "gripper":0}, # sleep
            # {"t": 400, "xyz": meet_xyz + np.array([-0.01, 0, 0.13]),          "quat": right_stir.elements,   "gripper":0}, # sleep
        ]
        self.random_values = random_values

class OpenLidPolicy(BasePolicy):
    language_instruction :str = 'open the lid of the cup' # 'use spoon to stir coffee'
    def generate_trajectory(self, ts_first, random_values:list=None):
        self.trajectory_generated = True
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        left_initial_loc = init_mocap_pose_left[:3]
        right_initial_loc = init_mocap_pose_right[:3]
        cup_info = np.array(ts_first.observation['env_state'])[:7]
        cup_xyz = cup_info[:3]
        cup_quat = cup_info[3:]

        lid_info = np.array(ts_first.observation['env_state'])[7:]
        lid_xyz = lid_info[:3].copy()
        lid_xyz[2] = 0
        lid_quat = lid_info[3:]


        # gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        # # gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=0)
        # # gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        # # gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        # gripper_stir_quat_right =  Quaternion(init_mocap_pose_right[3:]) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # meet_xyz = np.array([0.2, 0.7, 0.3])
        # lift_right = 0.00715

        left_initial_quat = Quaternion(np.array([1, 0 , 0, 0]))
        hold_deltas = [
            np.array([-0.0, 0, -0.1]),
            np.array([-0.0, 0, -0.1]),
            np.array([0.01, 0, -0.1]),
            np.array([+0.02, 0, -0.1]),
        ]
        stir_deltas = [
            np.array([-0.02, 0.01,     0.03]),
            np.array([-0.01, -0.01,    0.03]),
            np.array([0.01, 0.01,    0.03]),
            np.array([-0.01, -0.01,    0.04]),
            np.array([-0.02, 0.01,    0.04]),
            np.array([-0.02, 0.01,    0.05]),
            np.array([-0.01, -0.01,    0.06]),
        ]
        if random_values is None:
            meet_xyz = (cup_xyz + lid_xyz)/2 + np.random.uniform(-0.1, 0.1, 3)
            meet_xyz[2] = 0.3 + np.random.uniform(-0.1, 0.1)
            random_values = {
                "meet_xyz": meet_xyz,
            }
        else:
            logger.info(f"Using given random values.")
        # separate_xyz = meet_xyz + np.array([0.1, 0, 0.1])
        
        self.left_trajectory = [
            {"t": 0,    "xyz": left_initial_loc,                            "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 100,  "xyz": cup_xyz + np.array([-0.07, 0.0, 0.2]),        "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 125,  "xyz": cup_xyz + np.array([-0.03, 0.0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 150,  "xyz": cup_xyz +   np.array([-0.01, 0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 0}, 
            # {"t": 200,  "xyz": cup_xyz +   np.array([-0.01, 0, 0.05]),      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 200,  "xyz": random_values['meet_xyz'],    "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 300,  "xyz": random_values['meet_xyz'],      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 330,  "xyz": random_values['meet_xyz'],      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 360,  "xyz": random_values['meet_xyz'],      "quat": left_initial_quat.elements,            "gripper": 0}, 
            {"t": 400,  "xyz": random_values['meet_xyz'],    "quat": left_initial_quat.elements,            "gripper": 0}, 
        ]

        right_initial_quat = Quaternion(np.array([0, 0 , 0, -1]))
        right_down = right_initial_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)
        right_stir = right_initial_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-30)

        self.right_trajectory = [
            {"t": 0,   "xyz": right_initial_loc,            "quat": right_initial_quat.elements,          "gripper": 1}, # sleep
            {"t": 50, "xyz":  right_initial_loc,            "quat": right_initial_quat.elements,            "gripper": 1}, # sleep
            {"t": 100, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper": 0}, # sleep
            {"t": 150, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper": 0}, # sleep
            {"t": 200, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper": 0}, # sleep
            {"t": 250, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            {"t": 300, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            {"t": 350, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            {"t": 370, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            {"t": 390, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            {"t": 400, "xyz": right_initial_loc,            "quat": right_initial_quat.elements,            "gripper":0}, # sleep
            # {"t": 400, "xyz": meet_xyz + np.array([-0.01, 0, 0.13]),          "quat": right_stir.elements,   "gripper":0}, # sleep
        ]
        self.random_values = random_values


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

