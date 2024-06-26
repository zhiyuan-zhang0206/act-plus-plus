
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import h5py
import tensorflow
# use only cpu
tensorflow.config.set_visible_devices([], 'GPU')

START_FRAME = 200
START_FRAME -= 1
TIME_INTERVAL = 10
RIGHT_HAND_RELATIVE = True
language_embedding_path = Path(__file__).absolute().parent.parent / 'open_x_embodiment-main/models/string_to_embedding.npy'
language_to_embedding = np.load(language_embedding_path, allow_pickle=True).item()

def extract_index(name):
    return int(name.split('_')[-1])

def wxyz_to_xyzw(quat):
    if len(quat.shape) == 2:
        assert quat.shape[1] == 4
        return quat[:, [1,2,3,0]]
    else:
        return np.array([quat[1], quat[2], quat[3], quat[0]])

language_instructions = set()
episode_lengths = set()
def process_data(path, debug=False, absolute = False, right_hand_relative = False):
    # save_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(path.as_posix(), 'r') as root:
        language_instruction = root.attrs['language_instruction']
        language_instructions.add(language_instruction)
        episode_length = len(root['/observations/left_pose'])
        episode_lengths.add(episode_length)
        left_pose = root['/observations/left_pose'][()]
        right_pose = root['/observations/right_pose'][()]
        left_image = root['/observations/images/left_angle'][()]
        right_image = root['/observations/images/right_angle'][()]
        action = root['/action'][()]
        
        left_pose[:, 3:] = wxyz_to_xyzw(left_pose[:, 3:])
        right_pose[:, 3:] = wxyz_to_xyzw(right_pose[:, 3:])
        left_pose = left_pose[START_FRAME::TIME_INTERVAL]
        right_pose = right_pose[START_FRAME::TIME_INTERVAL]
        action = action[START_FRAME::TIME_INTERVAL][1:]
        left_image = left_image[START_FRAME::TIME_INTERVAL][:-1]
        right_image = right_image[START_FRAME::TIME_INTERVAL][:-1]

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

    left_location = left_pose[1:, :3]
    right_location = right_pose[1:, :3]
    left_rpy = R.from_quat(left_quaternion).as_euler('zyx')[1:]
    right_rpy = R.from_quat(right_quaternion).as_euler('zyx')[1:]
    
    # right hand as relative
    right_location_relative = (right_pose[:, :3] - left_pose[:, :3])[1:]
    right_quaternion_init = R.from_quat([0, 0, -1, 0])
    right_quaternion_relative = R.from_quat(right_quaternion) * right_quaternion_init.inv() * R.from_quat(left_quaternion).inv()
    right_rpy_relative = right_quaternion_relative.as_euler('zyx')[1:]
    # if out of range -pi/2, pi/2, then add or subtract pi
    # out_of_range_mask = np.abs(right_rpy_relative) > np.pi/2
    # right_rpy_relative[out_of_range_mask] = right_rpy_relative[out_of_range_mask] - np.pi * np.sign(right_rpy_relative[out_of_range_mask])
    # print(R.from_quat(right_quaternion[0]).as_quat())
    # print(right_quaternion_init.inv().as_quat())
    # print(R.from_quat(left_quaternion[0]).inv().as_quat())
    # print(R.from_quat(right_quaternion[0]) * right_quaternion_init.inv() * R.from_quat(left_quaternion[0]).inv())
    # R_final = R.from_quat(right_quaternion[0]) * right_quaternion_init.inv() * R.from_quat(left_quaternion[0]).inv()
    # print(R_final.as_quat())
    # print(R_final.as_matrix())
    # print(R_final.as_euler('zyx'))
    # print(right_rpy_relative[0])
    # breakpoint()
    # sys.exit()
    # right_location_relative_diff = np.diff(right_location_relative, axis=0)
    # right_rpy_relative_diff = R.from_euler('zyx', right_rpy_relative[1:]) * R.from_euler('zyx', right_rpy_relative[:-1]).inv()
    
    right_vector_diff = right_vector_diff if not right_hand_relative else right_location_relative
    rotation_delta_right = right_rpy_diff if not right_hand_relative else right_rpy_relative
    
    world_vector_left = left_location if absolute else left_vector_diff
    world_vector_right = right_location if absolute else right_vector_diff
    rotation_delta_left = left_rpy if absolute else left_rpy_diff
    rotation_delta_right = right_rpy if absolute else rotation_delta_right
    
    action_left = np.clip(1 - action[:, 6], 0, 1 )
    action_right = np.clip(1 - action[:, 13], 0, 1 )
    
    # save_path = save_dir / path.stem
    data = {
        'world_vector_left': world_vector_left,
        'rotation_delta_left': rotation_delta_left, #left_aa_diff,
        'world_vector_right': world_vector_right,
        'rotation_delta_right': rotation_delta_right,
        # 'world_vector_right_relative': right_location_relative,
        # 'rotation_delta_right_relative': right_rpy_relative.as_euler('zyx'),
        # 'world_vector_right_relative_diff': right_location_relative_diff,
        # 'rotation_delta_right_relative_diff': right_rpy_relative_diff.as_euler('zyx'),
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
    # np.save(save_path, data)
    return data

def test():
    hdf5_path = '/home/users/ghc/zzy/act-plus-plus/generated_data/stir/episode_0005.hdf5'
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
import os
from argparse import ArgumentParser
import json 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    raise ValueError('Boolean value expected.')

# python convert_data_format.py --right_hand_relative False --absolute True

def get_data_ranges(datas):
    world_vector_min = np.inf
    world_vector_max = -np.inf
    rotation_min = np.inf
    rotation_max = -np.inf
    for data in datas:
        world_vector_min = min(world_vector_min, min(np.min(data['world_vector_left']), np.min(data['world_vector_right'])))
        world_vector_max = max(world_vector_max, max(np.max(data['world_vector_left']), np.max(data['world_vector_right'])))
        rotation_min = min(rotation_min, min(np.min(data['rotation_delta_left']), np.min(data['rotation_delta_right'])))
        rotation_max = max(rotation_max, max(np.max(data['rotation_delta_left']), np.max(data['rotation_delta_right'])))
    return world_vector_min, world_vector_max, rotation_min, rotation_max

def normalize_location(datas):
    world_vector_left_list = []
    world_vector_right_list = []
    for data in datas:
        world_vector_left_list.append(data['world_vector_left'])
        world_vector_right_list.append(data['world_vector_right'])
    location_min = np.min(np.concatenate(world_vector_left_list + world_vector_right_list), axis=0)
    for data in datas:
        data['world_vector_left'] = data['world_vector_left'] - location_min
        data['world_vector_right'] = data['world_vector_right'] - location_min
    return location_min

def main(args):

    hdf5_directory = Path(__file__).absolute().parent / 'generated_data'
    data_config_path = Path(__file__).absolute().parent.parent / 'open_x_embodiment-main/data_config.json'
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    paths = []
    for path in sorted(list(hdf5_directory.rglob('*.hdf5'))):
        # if 'stir' in path.stem:
        if 'ACT' in path.as_posix():
            continue
        # if 'stir' not in path.as_posix():
        # if 'openlid' not in path.as_posix():
        # if 'transfercube' not in path.as_posix():
            continue
        paths.append(path)
    test = os.getenv('ZZY_DEBUG') == "True"
    if test:
        paths = paths[1:3]
    datas = []
    for i, p in tqdm(list(enumerate(paths))):
        data = process_data(p, debug= i==0, absolute=args.absolute, right_hand_relative=args.right_hand_relative)
        datas.append(data)
    
    world_vector_min, world_vector_max, rotation_min, rotation_max = get_data_ranges(datas)
    print(f"data ranges: {world_vector_min=}, {world_vector_max=}, {rotation_min=}, {rotation_max=}")
    
    if args.center_location:
        location_min = normalize_location(datas)
        data_config['location_min'] = location_min.tolist()
        with open(data_config_path, 'w') as f:
            json.dump(data_config, f)
        print(f"location min: {location_min}")
        
        world_vector_min, world_vector_max, rotation_min, rotation_max = get_data_ranges(datas)
        print(f"data ranges after centering: {world_vector_min=}, {world_vector_max=}, {rotation_min=}, {rotation_max=}")
    data_config["world_vector_min"] = world_vector_min.tolist()
    data_config["world_vector_max"] = world_vector_max.tolist()
    data_config["rotation_min"] = rotation_min.tolist()
    data_config["rotation_max"] = rotation_max.tolist()
    with open(data_config_path, 'w') as f:
        json.dump(data_config, f)
    save_dir = Path('/home/users/ghc/zzy/tensorflow-datasets/bimanual_zzy/data')
    save_dir.mkdir(exist_ok=True, parents=True)
    test_path = save_dir / 'test'
    train_path = save_dir / 'train'
    test_path.mkdir(exist_ok=True, parents=True)
    train_path.mkdir(exist_ok=True, parents=True)
    for i, data in tqdm(enumerate(datas), total=len(datas)):
        file_name = f'episode_{i:04d}.npy'
        if i % 10 == 0:
            np.save(test_path / file_name, data)
        else:
            np.save(train_path / file_name, data)
    print(f"language instructions: {language_instructions}")
    print(f"episode lengths: {episode_lengths}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--right_hand_relative', type=str2bool, required=True)
    parser.add_argument('--absolute', type=str2bool, required=True)
    parser.add_argument('--center_location', type=str2bool, required=True)
    args = parser.parse_args()
    main(args)
    # test()

# python convert_data_format.py --right_hand_relative False --absolute True --center_location True
