import time
import os
# os.environ['MUJOCO_GL'] = 'egl'
from typing import Literal
os.environ['MUJOCO_GL'] = 'osmesa'
import importlib
import warnings
from io import StringIO
class FailedToConvergeError(Exception):
    pass
class InterceptStdOut:
    def __init__(self, target):
        self.target = target
        self.buffer = StringIO()  # Using StringIO for more efficient string operations

    def write(self, message):
        self.buffer.write(message)  # Write to the buffer
        if "Failed to converge after" in message:
            print("Warning detected: ", message)
            raise FailedToConvergeError(message)
        self.target.write(message)

    def flush(self):
        self.buffer.flush()
        self.target.flush()
# warnings.filterwarnings("error", message=
# re.compile(r"Failed to converge after \d+ steps with norm \d+\.\d+"))
# import re
# warnings.filterwarnings("error", message=(r"Failed to converge after.*"))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
from scipy.spatial.transform import Slerp

class BimanualModelPolicy:
    def __init__(self, 
                 ckpt_path=None,
                 frame_interval:int=10,
                 dataset_debug:bool=True,
                 always_refresh:bool=False,
                 dropout_train:bool=False,
                 right_hand_relative:bool=False,
                ):
        instruction_to_embedding_path = Path('/home/users/ghc/zzy/open_x_embodiment-main/models/string_to_embedding.npy')
        self.instruction_to_embedding = np.load(instruction_to_embedding_path, allow_pickle=True).item()
        if ckpt_path is None:
            dataset_debug = True
        else:
            dataset_debug = False
        self.dataset_debug = dataset_debug
        self.frame_interval = frame_interval
        self.images_left = []
        self.images_right = []
        self.right_hand_relative = right_hand_relative  
        self.step = 0
        self.left_pose = None
        self.right_pose = None
        # return
        import sys
        sys.path.append('/home/users/ghc/zzy')
        rtx = importlib.import_module('open_x_embodiment-main' )
        self.batch_size = 14
        dataset = rtx.dataset.get_train_dataset(self.batch_size, bimanual=True, split='train[:1]', augmentation=False, shuffle=False)
        data = next(iter(dataset))
        self.action_storage = data['action']
        index = int(data['observation']['index'][self.batch_size-1, -1])
        logger.info(f"data episode {index=}")
        self.metadata = rtx.dataset.get_bimanual_dataset_episode_metadata()[index]
        
        if self.dataset_debug:
            pass
            # print(self.metadata['random_values'])
            # print(np.array(self.dataset_action['pose_right'])[:, :3])
            # import sys
            # sys.exit()
        else:
            logger.debug('loading model')
            # import rt1
            # from rt1_bimanual_inference_example import RT1BimanualPolicy
            model = rtx.models.rt1.BimanualRT1(
                num_image_tokens=81,
                num_action_tokens=11,
                layer_size=256,
                vocab_size=512,
                # Use token learner to reduce tokens per image to 81.
                use_token_learner=True,
                # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
                # world_vector_range=(-0.06, 0.06),
                world_vector_range=(-0.2, 0.2),
                )   
            self.policy = rtx.models.rt1_bimanual_inference_example.RT1BimanualPolicy(
                checkpoint_path=ckpt_path,
                model=model,
                seqlen=15,
                dropout_train = dropout_train
            )
            logger.debug('model loaded')
        self.always_refresh = always_refresh
    
    def set_language_instruction(self, language_instruction:str):
        
        # import tensorflow_hub as hub
        # embed = hub.load("/home/users/ghc/zzy/USE/ckpt_large_v5")
        self.language_instruction = language_instruction
        language_instruction_embedding = self.instruction_to_embedding[language_instruction]
        self.language_instruction_embedding = np.repeat(language_instruction_embedding[np.newaxis], 15, axis=0)
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
        
    def _calculate_new_pose(self, pose, delta, mode:Literal['left', 'right'], left_pose=None):
        if self.right_hand_relative and mode == 'right':
            assert left_pose is not None
            right_location = left_pose[:3] + delta[:3]
            right_quaternion = R.from_quat(pose[3:7]) * R.from_quat(left_pose[3:7]).inv()
            new_pose = np.concatenate([right_location, right_quaternion])
        else:
            new_location = pose[:3] + delta[:3]
            rpy_delta = R.from_euler('zyx', delta[3:6])
            current_rpy = R.from_quat(pose[3:7])
            new_quaternion = (rpy_delta * current_rpy).as_quat()
            # new_quaternion = R.from_euler('zyx', new_rpy).as_quat()
            new_pose = np.concatenate([new_location, new_quaternion])
            return new_pose
    
    def generate_action_buffer(self, current_pose, new_pose):
        diff = new_pose - current_pose
        actions = []
        start_quaternion = R.from_quat(current_pose[3:7])
        end_quaternion = R.from_quat(new_pose[3:7])
        times = np.linspace(0, 1, self.frame_interval, endpoint=False) + 1/self.frame_interval
        slerp = Slerp([0,1], R.concatenate([start_quaternion, end_quaternion]))
        interpolated_quaternions = slerp(times)
        for i in range(self.frame_interval):
            fraction = (i+1) / self.frame_interval
            action = current_pose + diff * fraction
            action[3:7] = interpolated_quaternions[i].as_quat()
            # action[3:7] /= np.linalg.norm(action[3:7])
            # action[11:15] /= np.linalg.norm(action[10:14])
            actions.append(action)
        return actions
    
    def action(self, policy_input):
        if self.dataset_debug:
            d = self.dataset_action
            left = np.concatenate([d['pose_left'].pop(0), d['gripper_left'].pop(0)], axis=-1)
            right = np.concatenate([d['pose_right'].pop(0), d['gripper_right'].pop(0)], axis=-1)
            return [left, right]
        else:
            detokenized = self.policy.action(policy_input)
            left = np.concatenate([detokenized['world_vector_left'], detokenized['rotation_delta_left'], np.clip(1-detokenized['gripper_closedness_action_left'], 0,1)], axis=0)
            right = np.concatenate([detokenized['world_vector_right'], detokenized['rotation_delta_right'], np.clip(1-detokenized['gripper_closedness_action_right'], 0,1)], axis=0)
            return left, right
    
    def __call__(self, ts):
        # action = np.zeros(14)
        observation = ts.observation
        # breakpoint()
        left, right = self._add_merge_image(observation['images']['left_angle'], observation['images']['right_angle'])
        if self.step % self.frame_interval == 0:
            policy_input = {
                "image_left": left,
                "image_right": right,
                "natural_language_embedding": self.language_instruction_embedding,
            }
            action = self.action(policy_input)
            if self.left_pose is None or self.always_refresh:
                self.left_pose = observation['left_pose']
                self.right_pose = observation['right_pose']
            self.left_pose = left_pose_new = self._calculate_new_pose( self.left_pose, action[0][:6], 'left')
            self.right_pose = right_pose_new = self._calculate_new_pose(self.right_pose, action[1][:6], 'right', left_pose=self.left_pose)
            left_gripper_new = np.zeros((1,))
            right_gripper_new = np.zeros((1,))
            current_pose = np.concatenate([observation['left_pose'], observation['qpos'][6:7],observation['right_pose'], observation['qpos'][13:14]])
            new_pose = np.concatenate([left_pose_new,  left_gripper_new, right_pose_new, right_gripper_new])
            self.action_buffer = self.generate_action_buffer(current_pose, new_pose)
        
        self.step += 1
        return self.action_buffer.pop(0)

    def reset(self, ):
        self.images_left = []
        self.images_right = []
        self.step = 0
        
        self.dataset_action = {'gripper_left':self.action_storage['gripper_closedness_action_left'][self.batch_size-1].numpy().tolist()[1:],
                                'gripper_right':self.action_storage['gripper_closedness_action_right'][self.batch_size-1].numpy().tolist()[1:],
                                'pose_left':np.concatenate([self.action_storage['world_vector_left'].numpy(), self.action_storage['rotation_delta_left'].numpy()], axis=-1)[self.batch_size-1].tolist()[1:],
                                'pose_right':np.concatenate([self.action_storage['world_vector_right'].numpy(),self.action_storage['rotation_delta_right'].numpy(), ], axis=-1)[self.batch_size-1].tolist()[1:]}

def make_action_q(observation):
    action_q = observation['qpos'].copy()
    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][0])
    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(observation['gripper_ctrl'][2])
    action_q[6] = left_ctrl
    action_q[6+7] = right_ctrl
    return action_q
    
from tqdm import trange
import random
random.seed(42)
import sys
def main(args):
    intercepted = InterceptStdOut(sys.stdout)
    sys.stdout = intercepted
    
    task_name = args['task_name']
    save_dir = args['save_dir']
    num_episodes = args['num_episodes']
    always_refresh = args['always_refresh']
    model_ckpt_path = args['model_ckpt_path']
    frame_interval = args['frame_interval']
    dropout_train = args['dropout_train']
    right_hand_relative = args['right_hand_relative']
    MODEL_POLICY_START_FRAME = 260
    # MODEL_POLICY_START_FRAME -= 1
    RENDER_START_FRAME = MODEL_POLICY_START_FRAME - 20
    assert MODEL_POLICY_START_FRAME >= RENDER_START_FRAME
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_stir_scripted':
        script_policy_cls = StirPolicy
    else:
        raise 
    # frame_interval = 20
    # model_ckpt_path = None
    # model_ckpt_path = '/home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-23_15-06-01/checkpoint_300'
    model_policy = BimanualModelPolicy(model_ckpt_path, frame_interval=frame_interval, always_refresh=always_refresh, dropout_train=dropout_train, right_hand_relative=right_hand_relative)
    model_policy.set_language_instruction(script_policy_cls.language_instruction)
    logger.info(f'language instruction: {script_policy_cls.language_instruction}')

    save_path = Path(os.path.join(save_dir, task_name, str(args['start_index'])))
    episode_idx = 0
    failed_times_limit = 10
    failed_times = 0
    while episode_idx < num_episodes:
        if failed_times >= failed_times_limit:
            logger.info('Failed too many times, stop.')
            break
        logger.info(f'Episode: {episode_idx+1}/{num_episodes}')
        model_policy.reset()
        script_policy = script_policy_cls()
        env_ee = make_ee_sim_env(task_name)
        # env_ee.task.object_start_pose = model_policy.metadata['objects_start_pose']
        ts_ee = env_ee.reset()
        # script_policy.generate_trajectory(ts_ee, model_policy.metadata['random_values'])
        script_policy.generate_trajectory(ts_ee)
        episode_ee = [ts_ee]
        
        object_info = env_ee.task.object_info
        
        env_q = make_sim_env(task_name, object_info=object_info)
        ts_q = env_q.reset()
        episode_q = [ts_q]
        episode_len = RENDER_START_FRAME + frame_interval * 20
        if episode_len >= 600:
            episode_len = 600
        policy = script_policy
        try:
            for step in trange(episode_len):
                # print(step)
                # if step >= 588:
                #     break
                if step == RENDER_START_FRAME:
                    env_q.task.start_render()
                # if step == MODEL_POLICY_START_FRAME:
                #     policy = model_policy
                # action_ee = policy(ts_q)
                # if step >= MODEL_POLICY_START_FRAME:
                # if step == 259:
                #     return
                if step >= 248:
                #     print(action_1:= model_policy(ts_q))
                #     print(action_2:= script_policy(ts_q))
                    action_1 = model_policy(ts_q)
                    # action_2 = script_policy(ts_q)
                    # action_ee = action_2
                    # print('dataset: ',action_1[:3])
                    # print('script:  ',action_2[8:11])
                    # print()
                    # if step % 20 > 10:
                    #     action_ee = action_1 
                    # else:
                    #     action_ee = action_2
                    # print('model: ', action_1[:3])
                    # print('script:', action_2[:3])
                    # diff = action_1 - action_2
                    action_ee = action_1
                    # action_ee = action_2
                    # print(f'{step=}')
                    # print('goal dataset', action_1[8:11])
                    # print('goal script ', action_2[8:11])
                    # print('diff', action_1[:3] - action_2[:3])
                    # print('diff', diff[:3], diff[8:11])
                    # action_ee[3:7] = action_1[3:7]
                    # action_ee[8:11] = action_1[8:11]
                    # action_ee[:3] = action_1[:3]
                    # action_ee[8:11] = action_1[8:11]
                else:
                    action_ee = script_policy(ts_q)
                try:
                    ts_ee = env_ee.step(action_ee)
                except FailedToConvergeError as e:  # Catching the warning as an exception
                    # if "Failed to converge" in str(e):
                    failed_times += 1
                    continue
                episode_ee.append(ts_ee)
                
                action_q = make_action_q(ts_ee.observation)
                try:
                    ts_q = env_q.step(action_q)
                except FailedToConvergeError as e:  # Catching the warning as an exception
                    # if "Failed to converge" in str(e):
                    failed_times += 1
                    continue
                # if 259 > step >= 248:
                #     print('actual      ', ts_q.observation['right_pose'][:3])
                #     print()
                episode_q.append(ts_q)
        except dm_control.rl.control.PhysicsError:
            logger.info('Physics error, continue.')
            failed_times += 1
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
    parser.add_argument('--always_refresh', action='store_true', default=False)
    parser.add_argument('--model_ckpt_path', action='store', type=str, required=True)
    parser.add_argument('--frame_interval', action='store', type=int, default=10)
    parser.add_argument('--dropout_train', action='store_true',default=False)
    parser.add_argument('--right_hand_relative', action='store_true', default=False)
    main(vars(parser.parse_args()))

