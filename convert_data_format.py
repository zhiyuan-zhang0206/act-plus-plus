import matplotlib.pyplot as plt
import sys

from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import h5py
from pyquaternion import Quaternion
START_FRAME = 250

language_embedding_path = Path(__file__).parent.parent / 'USE/string_to_embedding.npy'
language_to_embedding = np.load(language_embedding_path, allow_pickle=True).item()

def process_data(path, save_dir):
    save_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(path.as_posix(), 'r') as root:
        language_instruction = root.attrs['language_instruction']
        left_pose = root['/observations/left_pose'][()][START_FRAME:]
        left_vector = left_pose[:, :3]
        left_quaternion = left_pose[:, 3:]
        left_rpy = R.from_quat(left_quaternion).as_euler('zyx')
        # left_aa = np.zeros((left_quaternion.shape[0], 3))
        # for i in range(left_quaternion.shape[0]):
        #     left_aa[i] = Quaternion(left_quaternion[i]).axis * Quaternion(left_quaternion[i]).angle

        # left_aa_diff = np.diff(left_aa, axis=0)
        # left_aa_diff = np.concatenate((np.zeros((1, 3)), left_aa_diff), axis=0)

        right_pose = root['/observations/right_pose'][()][START_FRAME:]
        right_vector = right_pose[:, :3]
        right_quaternion = right_pose[:, 3:]
        right_rpy = R.from_quat(right_quaternion).as_euler('zyx')
        # right_aa = np.zeros((right_quaternion.shape[0], 3))
        # for i in range(right_quaternion.shape[0]):
        #     right_aa[i] = Quaternion(right_quaternion[i]).axis * Quaternion(right_quaternion[i]).angle
        
        # right_aa_diff = np.diff(right_aa, axis=0)
        # right_aa_diff = np.concatenate((np.zeros((1, 3)), right_aa_diff), axis=0)
        
        left_image = root['/observations/images/left_angle'][()][START_FRAME:]
        right_image = root['/observations/images/right_angle'][()][START_FRAME:]
        action = root['/action'][()][START_FRAME:]
        # in original data, 1 For open and 0 for close. Here we change to 1 for close and 0 for open
        action_left = np.clip(1 - action[:, 6], 0, 1 )
        action_right = np.clip(1 - action[:, 13], 0, 1 )
    
    save_path = save_dir / path.stem
    data = {
        'world_vector_left': left_vector,
        'rotation_delta_left': left_rpy, #left_aa_diff,
        'world_vector_right': right_vector,
        'rotation_delta_right': right_rpy, #right_aa_diff,
        'image_left': (np.clip(left_image, 0, 255) ).astype(np.uint8),
        'image_right': (np.clip(right_image, 0, 255) ).astype(np.uint8),
        'gripper_closedness_action_left': action_left,
        'gripper_closedness_action_right': action_right,
        'language_instruction': language_instruction,
        'language_embedding': language_to_embedding[language_instruction]
    }
    # save left_image as image
    # convert to uint8
    # left_image[0] = (np.clip(left_image[0], 0, 255) ).astype(np.uint8)
    # plt.imshow(left_image[0])
    # plt.savefig(save_path.with_suffix('.png'))
    
    # sys.exit()
    # plot left_aa_diff in 3 subplots
    # fig, axs = plt.subplots(3, 1)
    # for i in range(3):
    #     axs[i].plot(left_vector[:, i])
    # # save
    # print(save_path.as_posix())
    # plt.savefig(save_path.with_suffix('.png'))
    
    # sys.exit()
    np.save(save_path, data)
    # print shapes
    # print(left_vector.shape, left_aa_diff.shape, 
    #       right_vector.shape, right_aa_diff.shape,
    #         left_image.shape, right_image.shape, 
    #         action_left.shape, action_right.shape)
    # print(0)

from tqdm import tqdm
def main():
    hdf5_directory = Path(__file__).parent / 'generated_data' / 'sim_stir_scripted'
    save_dir = hdf5_directory.parent / 'processed_data'
    print(f'saving to {save_dir}')
    for p in tqdm(list(hdf5_directory.glob('*.hdf5'))):
        process_data(p, save_dir)
        
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

