import subprocess
from pathlib import Path
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

parrellel_number = 1
total_number = 50
overall_start_index = 0
dataset_path = Path('/home/users/ghc/zzy/act-plus-plus/generated_data/sim_stir_scripted')
if dataset_path.exists() and overall_start_index == 0:
    import shutil
    shutil.rmtree(dataset_path)

commands = []
# start_index = 0
for i in range(parrellel_number):
    each_num = total_number//parrellel_number
    start_index = i * each_num + overall_start_index
    command = f'python record_sim_episodes.py --num_episodes {each_num} --start_index {start_index}'
    commands.append(command)

processes = [run_command(command) for command in commands]

import time
t1 = time.time()
for process in processes:
    stdout, stderr = process.communicate()  # This waits for the process to finish
    print(stdout.decode(), stderr.decode())
    print('Time taken:', time.time() - t1)
