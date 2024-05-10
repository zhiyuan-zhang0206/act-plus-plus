import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
import h5py
from pyquaternion import Quaternion
START_FRAME = 200
START_FRAME_INDEX = START_FRAME 
camera_names = ['left_angle', 'right_angle']
def process_data(path:Path, save_dir:Path):
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / path.name
    with h5py.File(path.as_posix(), 'r') as root, h5py.File(save_path.as_posix(), 'w',) as new_root:
        for key, value in root.attrs.items():
            new_root.attrs[key] = value

        for key, value in root['random_values'].items():
            new_root.create_dataset(f'random_values/{key}', data=value)

        for key, value in root['observations'].items():
            if key == 'images':
                for cam_name, image in value.items():
                    new_root.create_dataset(f'observations/{key}/{cam_name}', data=image[START_FRAME_INDEX:])
            else:
                new_root.create_dataset(f'observations/{key}', data=value[START_FRAME_INDEX:])
        new_root.create_dataset('action', data=root['action'][START_FRAME_INDEX:])
        new_root.create_dataset('objects_start_pose', data=root['objects_start_pose'])

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
