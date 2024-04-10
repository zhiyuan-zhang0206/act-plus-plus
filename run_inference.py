import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['DISPLAY'] = 'egl'
# =:0
import numpy as np
import argparse
# import matplotlib.pyplot as plt
import h5py
import sys
sys.path.append('/home/users/ghc/zzy/open_x_embodiment-main/models')
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
def debug_visualize(ts, dataset_path, step):

    
    # save image
    # global image_idx
    # step = step
    if step % 20 != 0:
        return

    for key in ['left_angle', 'right_angle']:
        image = ts.observation['images'][key]
        image_path = dataset_path / f'{step}_{key}.png'
        plt.imsave(image_path.as_posix(), image)
    # image = ts.observation['images']['angle']
    # image_path = dataset_path / f'{step}_angle.png'
    # plt.imsave(image_path.as_posix(), image)
    # image = ts.observation['images']['front_close']
    # image_path = dataset_path / f'{step}_front_close.png'
    # plt.imsave(image_path.as_posix(), image)
    # logger.info(f'Saved to {image_path}')
    
    step += 1
    # if exit:
    #     import sys
    #     sys.exit()

def configure_logging(start_index, num_episodes):
    # add logger, save logs next to the dir "logs" next to this file, with file name as timestamp
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f'{timestamp}-index{start_index}.log'
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

class BimanualModelPolicy:
    def __init__(self, 
                 ckpt_path=None,
                ):
        instruction_to_embedding_path = Path('/home/users/ghc/zzy/USE/string_to_embedding.npy')
        self.instruction_to_embedding = np.load(instruction_to_embedding_path, allow_pickle=True).item()
        
        logger.debug('loading model')
        num_action_tokens = 11
        layer_size = 256
        vocab_size = 512
        sequence_length = 15
        num_image_tokens = 81
        import rt1
        from rt1_bimanual_inference_example import RT1BimanualPolicy
        model = rt1.BimanualRT1(
            num_image_tokens=num_image_tokens,
            num_action_tokens=num_action_tokens,
            layer_size=layer_size,
            vocab_size=vocab_size,
            # Use token learner to reduce tokens per image to 81.
            use_token_learner=True,
            # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
            world_vector_range=(-2.0, 2.0),
            )   
        self.policy = RT1BimanualPolicy(
            checkpoint_path=ckpt_path,
            model=model,
            seqlen=sequence_length,
        )
        logger.debug('model loaded')
    
    def set_language_instruction(self, language_instruction:str):
        
        # import tensorflow_hub as hub
        # embed = hub.load("/home/users/ghc/zzy/USE/ckpt_large_v5")
        self.language_instruction = language_instruction
        self.language_instruction_embedding = self.instruction_to_embedding[language_instruction]
        # self.language_instruction_embedding = embed([language_instruction])[0]
        # del embed
        
    def __call__(self, ts):
        # action = np.zeros(14)
        observation = ts.observation
        # breakpoint()
        policy_input = {
            "image_left": observation['images']['left_angle'],
            "image_right": observation['images']['right_angle'],
            "natural_language_embedding": self.language_instruction_embedding,
        }
        action = self.policy.action(policy_input)
        
        return action

def extract_q_pos(observation):
    action_q = observation['qpos']
    ctrl = observation['gripper_ctrl']
    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
    action_q[6] = left_ctrl
    action_q[6+7] = right_ctrl
    return action_q

from tqdm import trange
def main(args):
    start_index = args['start_index']
    logger.info(f'Start index: {start_index}')
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    configure_logging(start_index, num_episodes)
    # inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    ckpt_path = '/home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/coordinater+trans,data[:10]'
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    
    policy = BimanualModelPolicy(ckpt_path)
    policy.set_language_instruction(StirPolicy.language_instruction)
    # logger.info(f'Policy class: {policy_class.__name__}')
    # return
    success = []
    dataset_path = Path(os.path.join(dataset_dir, task_name))
    
    for episode_idx in range(num_episodes):
        logger.info(f'Episode: {episode_idx+1}/{num_episodes}')
        env_ee = make_ee_sim_env(task_name)
        ts_ee = env_ee.reset() 
        object_info = env_ee.task.object_info
        env_q = make_sim_env(task_name, object_info=object_info)
        episode_ee = [ts_ee]

        # last_action = None
        for step in trange(episode_len):
            action_q = extract_q_pos(ts_ee.observation)
            ts_q = env_q.step(action_q)
            action_ee = policy(ts_q)
            ts_ee = env_ee.step(action_ee)
            # episode_ee.append(ts_ee)

        # episode_return = np.sum([ts.reward for ts in episode[1:]])
        # episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        # if episode_max_reward == env_ee.task.max_reward:
        #     logger.info(f"{episode_idx=} Successful, {episode_return=}")
        # else:
        #     logger.info(f"{episode_idx=} Failed")

        joint_trajectory = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_trajectory = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_trajectory, gripper_ctrl_trajectory):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        # clear unused variables
        del env_ee
        del episode
        # del policy

        ts = env_q.reset()
        episode_replay = [ts]
        step=0
        for t in trange(len(joint_trajectory)): # note: this will increase episode length by 1
            action = joint_trajectory[t]
            ts = env_q.step(action)
            step+=1
            episode_replay.append(ts)


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
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_trajectory)
        # progress_bar = trange(max_timesteps)
        while joint_trajectory:
            action = joint_trajectory.pop(0)
            ts = episode_replay.pop(0)
            # ts = episode.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/left_pose'].append(ts.observation['left_pose'])
            data_dict['/observations/right_pose'].append(ts.observation['right_pose'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        data_path = dataset_path / f'episode_{episode_idx + start_index:04d}'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path = data_path.as_posix()
        # with h5py.File(data_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        #     root.attrs['sim'] = True
        #     # if hasattr(policy_class, 'language_instruction'):
        #     #     root.attrs['language_instruction'] = policy_class.language_instruction
        #     obs = root.create_group('observations')
        #     image = obs.create_group('images')
        #     for cam_name in camera_names:
        #         _ = image.create_dataset(cam_name, (max_timesteps, 300, 300, 3), dtype='uint8',
        #                                  chunks=(1, 300, 300, 3), )
        #     # compression='gzip',compression_opts=2,)
        #     # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        #     obs.create_dataset('qpos', (max_timesteps, 14))
        #     obs.create_dataset('qvel', (max_timesteps, 14))
        #     obs.create_dataset('left_pose', (max_timesteps, 7))
        #     obs.create_dataset('right_pose', (max_timesteps, 7))
        #     root.create_dataset('action', (max_timesteps, 14))

        #     for name, array in data_dict.items():
        #         root[name][...] = array

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

