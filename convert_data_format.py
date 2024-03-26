

from pathlib import Path
import numpy as np
import h5py

def convert():

    language_embedding_path = Path(__file__).parent.parent / 'USE/string_to_embedding.npy'
    language_to_embedding = np.load(language_embedding_path, allow_pickle=True).item()

    hdf5_directory = Path(__file__).parent / 'generated_data' / 'sim_stir_scripted'

    # read numpy file
    
    
def read(path):
    pass

convert()



