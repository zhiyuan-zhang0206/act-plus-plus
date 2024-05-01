
import os


def remove_all_file(directory, suffix):
    suffixes = []
    if isinstance(suffix, str):
        suffixes.append(suffix)
    else:
        suffixes = suffix
    for filename in os.listdir(directory):
        if any(filename.endswith(suffix) for suffix in suffixes):
            os.remove(os.path.join(directory, filename))

path = '/home/users/ghc/zzy/act-plus-plus/generated_data/stir'
remove_all_file(path, ['.zip', '.hdf5'])
