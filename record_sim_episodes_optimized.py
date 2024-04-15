import time
import os
# os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['DISPLAY'] = 'egl'
# =:0
import numpy as np
import argparse
# import matplotlib.pyplot as plt
import h5py
import dm_control
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env
from scripted_policy import StirPolicy
from pathlib import Path
from matplotlib import pyplot as plt
from loguru import logger
    
def make_action_q(observation):
    action_q = observation['qpos'].copy()
    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][0])
    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][2])
    action_q[6] = left_ctrl
    action_q[6+7] = right_ctrl
    return action_q
    
from tqdm import trange
def main(args):
    import random
    random.seed(1)
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """
    start_index = args['start_index']
    logger.info(f'Start index: {start_index}')
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_stir_scripted':
        policy_cls = StirPolicy
    else:
        raise NotImplementedError
    logger.info(f'Policy class: {policy_cls.__name__}')

    success = []
    dataset_path = Path(os.path.join(dataset_dir, task_name))
    episode_idx = 0
    while episode_idx < num_episodes:
        logger.info(f'Episode: {episode_idx+1}/{num_episodes}')

        script_policy = policy_cls(inject_noise)
        
        env_ee = make_ee_sim_env(task_name)
        ts_ee = env_ee.reset()
        episode_ee = [ts_ee]
        
        object_info = env_ee.task.object_info
        
        env_q = make_sim_env(task_name, object_info=object_info)
        ts_q = env_q.reset()
        episode_q = [ts_q]
        # episode_len = 400
        try:
            for step in trange(episode_len):
                action_ee = script_policy(ts_ee)
                ts_ee = env_ee.step(action_ee)
                episode_ee.append(ts_ee)
                
                if step == 250:
                    env_q.task.start_render()
                action_q = make_action_q(ts_ee.observation)
                ts_q = env_q.step(action_q)
                episode_q.append(ts_q)
        except dm_control.rl.control.PhysicsError:
            logger.info('Physics error, continue.')
            continue
        joint_trajectory = [ts_ee.observation['qpos'] for ts_ee in episode_ee]
        

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/left_pose': [],
            '/observations/right_pose': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_trajectory = joint_trajectory[:-1]
        # episode = episode[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_trajectory)
        print(f'max timesteps: {max_timesteps}')
        # progress_bar = trange(max_timesteps)
        while joint_trajectory:
            action = joint_trajectory.pop(0)
            ts_q = episode_q.pop(0)
            # ts = episode.pop(0)
            data_dict['/observations/qpos'].append(ts_q.observation['qpos'])
            data_dict['/observations/qvel'].append(ts_q.observation['qvel'])
            data_dict['/observations/left_pose'].append(ts_q.observation['left_pose'])
            data_dict['/observations/right_pose'].append(ts_q.observation['right_pose'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts_q.observation['images'][cam_name])

        # HDF5
        data_path = dataset_path / f'episode_{episode_idx + start_index:04d}'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path = data_path.as_posix()
        with h5py.File(data_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            if hasattr(policy_cls, 'language_instruction'):
                root.attrs['language_instruction'] = policy_cls.language_instruction
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 300, 300, 3), dtype='uint8',
                                         chunks=(1, 300, 300, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            obs.create_dataset('qpos', (max_timesteps, 14))
            obs.create_dataset('qvel', (max_timesteps, 14))
            obs.create_dataset('left_pose', (max_timesteps, 7))
            obs.create_dataset('right_pose', (max_timesteps, 7))
            root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        episode_idx += 1

    logger.info(f'Saved to {dataset_dir}')
    logger.info(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False, default='sim_stir_scripted')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=False, default=(Path(__file__).parent / 'generated_data').as_posix())
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False, default=1)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--start_index', action='store', type=int, help='start_index', required=False, default=0)
    main(vars(parser.parse_args()))

