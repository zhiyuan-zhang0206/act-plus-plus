import time
import os
# os.environ['MUJOCO_GL'] = 'egl'
if __name__ == '__main__':
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
from scripted_policy import StirPolicy, OpenLidPolicy
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import trange
from loguru import logger

def assemble_image_for_oncreen_render(ts):
    image_left = ts.observation['images']['left_angle']
    image_right = ts.observation['images']['right_angle']
    image = np.concatenate([image_left, image_right], axis=1)
    return image
def configure_logging():
    # add logger, save logs next to the dir "logs" next to this file, with file name as timestamp
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f'{timestamp}.log'
    logger.add(log_file.as_posix())

task_name_to_script_policy_cls = {
    'stir': StirPolicy,
    'openlid': OpenLidPolicy,
}
def make_action_q(observation):
    action_q = observation['qpos'].copy()
    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][0])
    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][2])
    action_q[6] = left_ctrl
    action_q[6+7] = right_ctrl
    return action_q
    
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
    configure_logging()
    start_index = args['start_index']
    logger.info(f'Start index: {start_index}')
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    render_start = args['render_start']
    render_interval = args['render_interval']
    inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name not in task_name_to_script_policy_cls:
        raise ValueError(f'Unsupported task name: {task_name}')
    policy_cls = task_name_to_script_policy_cls[task_name]
    
    logger.info(f'Policy class: {policy_cls.__name__}')

    dataset_path = Path(os.path.join(dataset_dir, task_name))
    episode_idx = 0
    final_rewards = []
    while episode_idx < num_episodes:
        logger.info(f'Episode: {episode_idx+1}/{num_episodes}')

        script_policy = policy_cls(inject_noise)
        
        env_ee = make_ee_sim_env(task_name)
        ts_ee = env_ee.reset()
        episode_ee = [ts_ee]
        
        object_info = env_ee.task.object_info
        script_policy.generate_trajectory(ts_ee)
        random_values = script_policy.random_values
        objects_start_pose = env_ee.task.objects_start_pose
        env_q = make_sim_env(task_name, object_info=object_info)
        ts_q = env_q.reset()
        episode_q = [ts_q]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(assemble_image_for_oncreen_render(ts_q))
            plt.ion()
        try:
            for step in trange(episode_len):
                if onscreen_render:
                    plt_img.set_data(assemble_image_for_oncreen_render(ts_q))
                    plt.pause(0.0001)
                action_ee = script_policy(ts_ee)
                ts_ee = env_ee.step(action_ee)
                episode_ee.append(ts_ee)
                
                if step == 0:
                    env_q.task.set_render_state(False)
                elif step == render_start:
                    env_q.task.set_render_state(True)
                
                if step % render_interval == 0 and step >= render_start:
                    env_q.task.set_render_state(True)
                else:
                    env_q.task.set_render_state(False)
                    
                action_q = make_action_q(ts_ee.observation)
                ts_q = env_q.step(action_q)
                episode_q.append(ts_q)
        except dm_control.rl.control.PhysicsError:
            logger.info('Physics error, continue.')
            plt.close()
            continue
        plt.close()
        joint_trajectory = [ts_ee.observation['qpos'] for ts_ee in episode_ee]
        episode_return = np.sum([ts.reward for ts in episode_q[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_q[1:]])
        final_reward = episode_q[-1].reward
        final_rewards.append(final_reward)
        logger.info(f'Episode return: {episode_return}, max reward: {episode_max_reward}, final reward: {final_reward}')

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

        # examine data
        left_diffs = []
        right_diffs = []
        for i in range(len(data_dict['/observations/left_pose'][render_start::render_interval][:-1])):
            location_left = data_dict['/observations/left_pose'][render_start::render_interval][i]
            location_left_next = data_dict['/observations/left_pose'][render_start::render_interval][i+1]
            location_right = data_dict['/observations/right_pose'][render_start::render_interval][i]
            location_right_next = data_dict['/observations/right_pose'][render_start::render_interval][i+1]
            location_left_diff = location_left_next - location_left
            location_right_diff = location_right_next - location_right
            left_diffs.append(location_left_diff)
            right_diffs.append(location_right_diff)
        left_diff_max = np.max(np.abs(np.array(left_diffs)))
        right_diff_max = np.max(np.abs(np.array(right_diffs)))
        logger.info(f'Left diff max: {left_diff_max}')
        logger.info(f'Right diff max: {right_diff_max}')
        if left_diff_max > 0.1 or right_diff_max > 0.1:
            logger.critical('Large diff in location!')

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
            for key, value in random_values.items():
                root.create_dataset(f'random_values/{key}', data=value)
            root.create_dataset('objects_start_pose', data=objects_start_pose)
        episode_idx += 1
    logger.info(f'Max final rewards: {np.max(final_rewards)}, mean final rewards: {np.mean(final_rewards)}, min final rewards: {np.min(final_rewards)}')
    logger.info(f'Saved to {dataset_dir}')
    # logger.info(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False, default='stir')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=False, default=(Path(__file__).parent / 'generated_data').as_posix())
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False, default=1)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--start_index', action='store', type=int, help='start_index', required=False, default=0)
    parser.add_argument('--render_start', action='store', type=int, help='render_start', required=False, default=250)
    parser.add_argument('--render_interval', action='store', type=int, help='render_interval', required=False, default=1)
    main(vars(parser.parse_args()))

# python record_sim_episodes_optimized.py --task_name stir --dataset_dir generated_data/stir --onscreen_render
# python record_sim_episodes_optimized.py --task_name openlid --dataset_dir generated_data/openlid --onscreen_render

# python visualize_episodes.py --dataset_dir generated_data/stir
# python visualize_episodes.py --dataset_dir generated_data/openlid

