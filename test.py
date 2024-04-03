from pathlib import Path
path = Path('/home/users/ghc/zzy/tensorflow-datasets/bimanua_zzy/data/train')
path_file = Path("/home/users/ghc/zzy/tensorflow-datasets/bimanual_zzy/data/train/episode_0000.npy")

# print(path.is_absolute())
# for f in path.is_absolute().glob("*.npy"):
#     print(f)
# for f in path_file.parent.glob("*.npy"):
#     print(f)
print(path_file.parent)
print(path)