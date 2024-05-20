from pathlib import Path
import shutil

for file in Path(__file__).absolute().parent.glob('*.xml'):
    # print(file.name)
    if '_gen1' in file.name:
        # copy file and name it
        print(file)
        new_name = file.name.replace('_gen1', '_gen0')
        shutil.copy(file, file.parent / new_name)
