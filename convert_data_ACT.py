import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import h5py
from pyquaternion import Quaternion
START_FRAME = 200
START_FRAME_INDEX = START_FRAME - 1

def process_data(path:Path, save_dir:Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(path.as_posix(), 'r') as root:
        language_instruction = root.attrs['language_instruction']
        left_pose = root['/observations/left_pose'][()][START_FRAME_INDEX:]
        right_pose = root['/observations/right_pose'][()][START_FRAME_INDEX:]
        left_image = root['/observations/images/left_angle'][()][START_FRAME_INDEX:]
        right_image = root['/observations/images/right_angle'][()][START_FRAME_INDEX:]
        action = root['/action'][()][START_FRAME_INDEX:]
        
        random_values = {}
        for key, dataset in root['random_values'].items():
            random_values[key] = dataset[()]
        objects_start_pose = root['objects_start_pose'][()]
    
    save_path = save_dir / f'{path.name}'
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('language_instruction', data=language_instruction)
        f.create_dataset('left_pose', data=left_pose)
        f.create_dataset('right_pose', data=right_pose)
        f.create_dataset('left_image', data=left_image)
        f.create_dataset('right_image', data=right_image)
        f.create_dataset('action', data=action)
        for key, value in random_values.items():
            f.create_dataset(f'random_values/{key}', data=value)
        f.create_dataset('objects_start_pose', data=objects_start_pose)

def main():
    root_directory = Path(__file__).parent / 'generated_data'
    # from_directory = Path(__file__).parent / 'generated_data' / 'stir'
    # to_directory = Path(__file__).parent / 'generated_data' / 'stir_ACT'
    for path in root_directory.glob('*'):
        if 'ACT' in path.stem:
            continue
        else:
            from_directory = path
            to_directory = path.parent / (path.stem + '_ACT')
            to_directory.mkdir(exist_ok=True)

            paths = sorted(list(from_directory.glob('*.hdf5')))

            for i, p in tqdm(list(enumerate(paths))):
                process_data(p, to_directory)

if __name__ == '__main__':
    main()
    # test()

# python convert_data_format.py --right_hand_relative False --absolute True
