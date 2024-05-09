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
    with h5py.File(path.as_posix(), 'r') as root:
        left_pose = root['/observations/left_pose'][()][START_FRAME_INDEX:]
        right_pose = root['/observations/right_pose'][()][START_FRAME_INDEX:]
        left_image = root['/observations/images/left_angle'][()][START_FRAME_INDEX:]
        right_image = root['/observations/images/right_angle'][()][START_FRAME_INDEX:]
        action = root['/action'][()][START_FRAME_INDEX:]

        save_path = save_dir / f'{path.name}'
        with h5py.File(save_path, 'w') as new_root:
            # Copy datasets that are unchanged
            for key in root.keys():
                if key not in ['observations/left_pose', 'observations/right_pose', 'observations/images/left_angle', 'observations/images/right_angle', 'action']:
                    new_root.copy(root[key], key)

            # Write or overwrite modified datasets
            def create_or_replace(group, name, data):
                if name in group:
                    del group[name]
                group.create_dataset(name, data=data)

            create_or_replace(new_root, 'observations/left_pose', left_pose)
            create_or_replace(new_root, 'observations/right_pose', right_pose)
            create_or_replace(new_root, 'observations/images/left_angle', left_image)
            create_or_replace(new_root, 'observations/images/right_angle', right_image)
            create_or_replace(new_root, 'action', action)
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
