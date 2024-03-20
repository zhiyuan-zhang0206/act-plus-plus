

from pathlib import Path
import numpy as np
import h5py

def convert():

    language_embedding_path = Path(__file__).parent.parent / 'USE/string_to_embedding.npy'
    language_to_embedding = np.load(language_embedding_path, allow_pickle=True).item()

    hdf5_directory = Path(__file__).parent / 'generated_data' / 'sim_stir_scripted'

    # read hdf5 file
    for dataset_path in hdf5_directory.glob('*.hdf5'):
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            image_dict = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
            language_instruction = root.attrs['language_instruction']

        language_embedding = language_to_embedding[language_instruction]
    
def read(path):
    pass





