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
from scipy.spatial.transform import Rotation as R

class BimanualModelPolicy:
    def __init__(self, 
                 ckpt_path=None,
                 frame_interval:int=10
                ):
        instruction_to_embedding_path = Path('/home/users/ghc/zzy/USE/string_to_embedding.npy')
        self.instruction_to_embedding = np.load(instruction_to_embedding_path, allow_pickle=True).item()
        self.frame_interval = frame_interval
        return
        logger.debug('loading model')
        num_action_tokens = 11
        layer_size = 256
        vocab_size = 32
        sequence_length = 15
        num_image_tokens = 81
        import sys
        sys.path.append('/home/users/ghc/zzy/open_x_embodiment-main/models')
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
            world_vector_range=(-0.06, 0.06),
            )   
        self.policy = RT1BimanualPolicy(
            checkpoint_path=ckpt_path,
            model=model,
            seqlen=sequence_length,
        )
        logger.debug('model loaded')
        self.images_left = []
        self.images_right = []
        self.step = 0
    
    def set_language_instruction(self, language_instruction:str):
        
        # import tensorflow_hub as hub
        # embed = hub.load("/home/users/ghc/zzy/USE/ckpt_large_v5")
        self.language_instruction = language_instruction
        self.language_instruction_embedding = self.instruction_to_embedding[language_instruction]
        # self.language_instruction_embedding = embed([language_instruction])[0]
        # del embed
        
    def _add_merge_image(self, image_left, image_right):
        self.images_left.append(image_left)
        self.images_right.append(image_right)
        if len(self.images_left) > 15:
            self.images_left.pop(0)
            self.images_right.pop(0)
        assert len(self.images_left) == len(self.images_right)
        left_input = np.stack(self.images_left, axis=0)
        left_input_padded = np.zeros((15, 300, 300, 3))
        left_input_padded[-left_input.shape[0]:] = left_input
        right_input = np.stack(self.images_right, axis=0)
        right_input_padded = np.zeros((15, 300, 300, 3))
        right_input_padded[-right_input.shape[0]:] = right_input
        return left_input_padded, right_input_padded
        
    def _calculate_new_pose(self, pose, delta):
        new_location = pose[:3] + delta[:3]
        rpy_delta = delta[3:6]
        current_rpy = R.from_quat(pose[3:7]).as_euler('zyx')
        new_rpy = current_rpy + rpy_delta
        new_quaternion = R.from_euler('zyx', new_rpy).as_quat()
        return np.concatenate([new_location, new_quaternion])
    
    def generate_action_buffer(self, current_pose, new_pose):
        diff = new_pose - current_pose
        actions = []
        for i in range(self.frame_interval):
            fraction = (i+1) / self.frame_interval
            action = current_pose + diff * fraction
            actions.append(action)
        return actions
            
    def __call__(self, ts):
        # action = np.zeros(14)
        observation = ts.observation
        breakpoint()
        left, right = self._add_merge_image(observation['images']['left_angle'], observation['images']['right_angle'])
        if self.step % self.frame_interval:
            policy_input = {
                "image_left": left,
                "image_right": right,
                "natural_language_embedding": self.language_instruction_embedding,
            }
            action = self.policy.action(policy_input)
            left_pose_new = self._calculate_new_pose(observation['left_pose'], action[0])
            right_pose_new = self._calculate_new_pose(observation['right_pose'], action[1])
            current_pose = np.concatenate([observation['left_pose'], observation['right_pose']])
            new_pose = np.concatenate([left_pose_new, right_pose_new])
            self.action_buffer = self.generate_action_buffer(current_pose, new_pose)
        
        self.step += 1
        return self.action_buffer.pop(0)

    def reset(self, ):
        self.images_left = []
        self.images_right = []
        self.step = 0

def make_action_q(observation):
    action_q = observation['qpos'].copy()
    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][0])
    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][2])
    action_q[6] = left_ctrl
    action_q[6+7] = right_ctrl
    return action_q
    
from tqdm import trange
import random
random.seed(1)
def main(args):
    
    task_name = args['task_name']
    save_dir = args['save_dir']
    num_episodes = args['num_episodes']
    # inject_noise = False
    RENDER_START_FRAME = 250
    MODEL_POLICY_START_FRAME = 260
    assert MODEL_POLICY_START_FRAME >= RENDER_START_FRAME
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_stir_scripted':
        script_policy_cls = StirPolicy
    else:
        raise NotImplementedError
    model_ckpt_path = '/home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/coordinater-train[:1]-1000-2024-04-16_15-56-39'
    model_policy = BimanualModelPolicy(model_ckpt_path)
    model_policy.set_language_instruction(script_policy_cls.language_instruction)
    logger.info(f'language instruction: {script_policy_cls.language_instruction}')

    save_path = Path(os.path.join(save_dir, task_name))
    episode_idx = 0
    while episode_idx < num_episodes:
        logger.info(f'Episode: {episode_idx+1}/{num_episodes}')
        model_policy.reset()
        script_policy = script_policy_cls()
        env_ee = make_ee_sim_env(task_name)
        ts_ee = env_ee.reset()
        script_policy.generate_trajectory(ts_ee)
        episode_ee = [ts_ee]
        
        object_info = env_ee.task.object_info
        
        env_q = make_sim_env(task_name, object_info=object_info)
        ts_q = env_q.reset()
        episode_q = [ts_q]
        
        policy = script_policy
        try:
            for step in trange(episode_len):
                if step == RENDER_START_FRAME:
                    env_q.task.start_render()
                if step == MODEL_POLICY_START_FRAME:
                    # policy = model_policy
                    pass
                action_ee = policy(ts_q)
                ts_ee = env_ee.step(action_ee)
                episode_ee.append(ts_ee)
                
                action_q = make_action_q(ts_ee.observation)
                ts_q = env_q.step(action_q)
                episode_q.append(ts_q)
        except dm_control.rl.control.PhysicsError:
            logger.info('Physics error, continue.')
            continue
        joint_trajectory = [ts_ee.observation['qpos'] for ts_ee in episode_ee]

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/left_pose': [],
            '/observations/right_pose': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_trajectory = joint_trajectory[:-1]
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
        data_path = save_path / f'episode_{episode_idx}'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path = data_path.as_posix()
        with h5py.File(data_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            if hasattr(script_policy_cls, 'language_instruction'):
                root.attrs['language_instruction'] = script_policy_cls.language_instruction
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

    logger.info(f'Saved to {save_dir}')
    # logger.info(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False, default='sim_stir_scripted')
    parser.add_argument('--save_dir', action='store', type=str, help='dataset saving dir', required=False, default=(Path(__file__).parent / 'evaluation_data').as_posix())
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False, default=1)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--start_index', action='store', type=int, help='start_index', required=False, default=0)
    main(vars(parser.parse_args()))

