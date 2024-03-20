import time
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import argparse
# import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy, StirPolicy
from pathlib import Path
from matplotlib import pyplot as plt
from loguru import logger
# import IPython
# e = IPython.embed

# python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir /root/autodl-tmp/act-plus-plus/generated_data --num_episodes 10
# python3 record_sim_episodes.py --task_name sim_stir_scripted --dataset_dir /root/autodl-tmp/act-plus-plus/generated_data --num_episodes 10

image_idx = 0
debug = False
def debug_visualize_exit(ts, dataset_path, exit=False):
    if not debug:
        return
    
    # save image
    global image_idx

    image = ts.observation['images']['top']
    image_path = dataset_path / f'{image_idx}_top.png'
    plt.imsave(image_path.as_posix(), image)
    image = ts.observation['images']['angle']
    image_path = dataset_path / f'{image_idx}_angle.png'
    plt.imsave(image_path.as_posix(), image)
    image = ts.observation['images']['front_close']
    image_path = dataset_path / f'{image_idx}_front_close.png'
    plt.imsave(image_path.as_posix(), image)
    print(f'Saved to {image_path}')
    
    image_idx += 1
    if exit:
        import sys
        sys.exit()

def configure_logging():
    # add logger, save logs next to the dir "logs" next to this file, with file name as timestamp
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f'{timestamp}.log'
    logger.add(log_file.as_posix())

def log_qpos(qpos):
    logger.info('QPose')
    logger.info(f"Left : {qpos[6]: .3f}, q: {' '.join(f'{i: .3f}' for i in qpos[0:7])}")
    logger.info(f"Right: {qpos[13]: .3f}, q: {' '.join(f'{i: .3f}' for i in qpos[6:13])}")

def log_action(action):
    # 1 means open and 0 means close
    logger.info('Action')
    logger.info(f"Left : {action[7]: .3f}, xyz: {' '.join(f'{i: .3f}' for i in action[0:3])}, quat: {' '.join(f'{i: .3f}' for i in action[3:7])}")
    logger.info(f"Right: {action[15]: .3f}, xyz: {' '.join(f'{i: .3f}' for i in action[7:10])}, quat: {' '.join(f'{i: .3f}' for i in action[10:14])}")

def log_env_state(state):
    logger.info('Env state')
    logger.info(f"Cup   : {' '.join(f'{i: .3f}' for i in state[0:3])}, {' '.join(f'{i: .3f}' for i in state[3:7])}")
    logger.info(f"Spoon : {' '.join(f'{i: .3f}' for i in state[7:10])}, {' '.join(f'{i: .3f}' for i in state[10:14])}")

def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """
    configure_logging()

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    # onscreen_render = args['onscreen_render']
    inject_noise = False
    # render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_stir_scripted':
        policy_cls = StirPolicy
    else:
        raise NotImplementedError
    print(f'Policy class: {policy_cls.__name__}')

    success = []
    dataset_path = Path(os.path.join(dataset_dir, task_name))
    # clear this directory
    for file in dataset_path.glob('*'):
        os.remove(file)
    for episode_idx in range(num_episodes):
        print(f'Episode: {episode_idx+1}/{num_episodes}')

        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)

        # last_action = None
        for step in range(episode_len):
            logger.info(f"Step: {step+1}/{episode_len}")
            log_env_state(ts.observation['env_state'])
            log_qpos(ts.observation['qpos'])
            action = policy(ts)
            log_action(action)
            # if last_action is not None:
                # print(action == last_action)
                # pass
            ts = env.step(action)
            if step % 1 == 0:
                # print(f"Step: {step}/{episode_len}")
                debug_visualize_exit(ts, dataset_path, exit = step == 250)
            episode.append(ts)
            # last_action = action
        debug_visualize_exit(ts, dataset_path, exit=True)

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]

        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        # plt.close()

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
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        # t0 = time.time()
        data_path = dataset_path / f'episode_{episode_idx}'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path = data_path.as_posix()
        with h5py.File(data_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            if 'language_instruction' in root.attrs:
                root.attrs['language_instruction'] = policy_cls.language_instruction
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            # qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            # qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            # action = root.create_dataset('action', (max_timesteps, 14))
            obs.create_dataset('qpos', (max_timesteps, 14))
            obs.create_dataset('qvel', (max_timesteps, 14))
            root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        # print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False, default='sim_stir_scripted')
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=False, default=(Path(__file__).parent / 'generated_data').as_posix())
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False, default=5)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

