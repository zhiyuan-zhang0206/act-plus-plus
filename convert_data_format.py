import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import h5py
from pyquaternion import Quaternion
START_FRAME = 260
START_FRAME -= 1
TIME_INTERVAL = 10
language_embedding_path = Path(__file__).parent.parent / 'open_x_embodiment-main/models/string_to_embedding.npy'
language_to_embedding = np.load(language_embedding_path, allow_pickle=True).item()

def extract_index(name):
    return int(name.split('_')[-1])

LOCATION_MAX = 0
LOCATION_MIN = 0
ROTATION_MAX = 0
ROTATION_MIN = 0
def update_max_min(l1, l2, r1, r2):
    location_max = np.max(np.concatenate([l1, l2]))
    location_min = np.min(np.concatenate([l1, l2]))
    rotation_max = np.max(np.concatenate([r1, r2]))
    rotation_min = np.min(np.concatenate([r1, r2]))
    global LOCATION_MAX, LOCATION_MIN, ROTATION_MAX, ROTATION_MIN
    LOCATION_MAX = max(LOCATION_MAX, location_max)
    LOCATION_MIN = min(LOCATION_MIN, location_min)
    ROTATION_MAX = max(ROTATION_MAX, rotation_max)
    ROTATION_MIN = min(ROTATION_MIN, rotation_min)

def process_data(path, save_dir, debug=False):
    save_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(path.as_posix(), 'r') as root:
        language_instruction = root.attrs['language_instruction']
        left_pose = root['/observations/left_pose'][()][START_FRAME-TIME_INTERVAL::TIME_INTERVAL]
        right_pose = root['/observations/right_pose'][()][START_FRAME-TIME_INTERVAL::TIME_INTERVAL]
        left_image = root['/observations/images/left_angle'][()][START_FRAME::TIME_INTERVAL]
        right_image = root['/observations/images/right_angle'][()][START_FRAME::TIME_INTERVAL]
        action = root['/action'][()][START_FRAME::TIME_INTERVAL]
        
        random_values = {}
        for key, dataset in root['random_values'].items():
            random_values[key] = dataset[()]
        objects_start_pose = root['objects_start_pose'][()]
        
    left_vector_diff = np.diff(left_pose[:, :3], axis=0)
    left_quaternion = left_pose[:, 3:] / np.linalg.norm(left_pose[:, 3:], axis=1, keepdims=True)
    left_quaternion_diffs = R.from_quat(left_quaternion[1:]) * R.from_quat(left_quaternion[:-1]).inv()
    left_rpy_diff = left_quaternion_diffs.as_euler('zyx')
    right_vector_diff = np.diff(right_pose[:, :3], axis=0)
    right_quaternion = right_pose[:, 3:] / np.linalg.norm(right_pose[:, 3:], axis=1, keepdims=True)
    right_quaternion_diffs = R.from_quat(right_quaternion[1:]) * R.from_quat(right_quaternion[:-1]).inv()
    right_rpy_diff = right_quaternion_diffs.as_euler('zyx')
    
    # right hand as relative
    right_location_relative = right_pose[:, :3] - left_pose[:, :3]
    right_rpy_relative = (R.from_quat(right_quaternion) * R.from_quat(left_quaternion).inv()).as_euler('zyx')
    right_location_relative_diff = np.diff(right_location_relative, axis=0)
    right_rpy_relative_diff = R.from_euler('zyx', right_rpy_relative[1:]) * R.from_euler('zyx', right_rpy_relative[:-1]).inv()
    
    
    update_max_min(left_vector_diff, right_vector_diff, left_rpy_diff, right_rpy_diff)
    action_left = np.clip(1 - action[:, 6], 0, 1 )
    action_right = np.clip(1 - action[:, 13], 0, 1 )
    
    save_path = save_dir / path.stem
    data = {
        'world_vector_left': left_vector_diff,
        'rotation_delta_left': left_rpy_diff, #left_aa_diff,
        'world_vector_right': right_vector_diff,
        'world_vector_right_relative': right_location_relative,
        'rotation_delta_right_relative': right_rpy_relative.as_euler('zyx'),
        'world_vector_right_relative_diff': right_location_relative_diff,
        'rotation_delta_right_relative_diff': right_rpy_relative_diff.as_euler('zyx'),
        'rotation_delta_right': right_rpy_diff, #right_aa_diff,
        'image_left': (np.clip(left_image, 0, 255) ).astype(np.uint8),
        'image_right': (np.clip(right_image, 0, 255) ).astype(np.uint8),
        'gripper_closedness_action_left': action_left,
        'gripper_closedness_action_right': action_right,
        'language_instruction': language_instruction,
        'language_embedding': language_to_embedding[language_instruction],
        "index": extract_index(path.stem),
        'random_values': random_values,
        'objects_start_pose': objects_start_pose,
    }
    if debug:
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape)
    np.save(save_path, data)

def test():
    hdf5_path = '/home/users/ghc/zzy/act-plus-plus/generated_data/sim_stir_scripted/episode_0005.hdf5'
    with h5py.File(hdf5_path, 'r') as root:        
        left_pose = root['/observations/left_pose'][()]  
        right_pose = root['/observations/right_pose'][()]# [START_FRAME-TIME_INTERVAL::TIME_INTERVAL]
        random_values = {}
        for key, dataset in root['random_values'].items():
            random_values[key] = dataset[()]
    print(right_pose[329:340,:3])
    right_pose = right_pose[START_FRAME-TIME_INTERVAL::TIME_INTERVAL]
    # print(right_pose[:2,:3])
    right_vector_diff = np.diff(right_pose[:, :3], axis=0)
    # print(right_pose[1:, :3] - right_pose[:-1, :3])
    print(right_vector_diff[0])
    # print(random_values)

def main():
    hdf5_directory = Path(__file__).parent / 'generated_data' / 'sim_stir_scripted'
    save_dir = hdf5_directory.parent / 'processed_data'
    print(f'saving to {save_dir}')
    paths = sorted(list(hdf5_directory.glob('*.hdf5')))
    # process_data(paths[0], save_dir, debug=True)
    for i, p in tqdm(list(enumerate(paths))):
        process_data(p, save_dir, debug= i==0)
    print(f"data ranges: {LOCATION_MIN=}, {LOCATION_MAX=}, {ROTATION_MIN=}, {ROTATION_MAX=}")
    # select 10% and put into 'test', else 'train'
    train_path = save_dir / 'train'
    test_path = save_dir / 'test'
    train_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)
    for i, p in enumerate(save_dir.glob('*.npy')):
        if i % 10 == 0:
            p.rename(test_path / p.name)
        else:
            p.rename(train_path / p.name)

    # all move to tf datasets
    ds_dir = Path('/home/users/ghc/zzy/tensorflow-datasets/bimanual_zzy/data')
    ds_dir.mkdir(exist_ok=True, parents=True)
    for p in train_path.glob('*.npy'):
        
        (ds_dir/'train').mkdir(exist_ok=True, parents=True)
        p.rename(ds_dir / 'train' / p.name)
    for p in test_path.glob('*.npy'):
        
        (ds_dir/'test').mkdir(exist_ok=True, parents=True)
        p.rename(ds_dir / 'test' / p.name)

if __name__ == '__main__':
    main()
    # test()

