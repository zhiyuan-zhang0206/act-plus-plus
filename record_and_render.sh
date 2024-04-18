python record_sim_episodes_with_model.py \
    --task_name sim_stir_scripted \
    --num_episodes 1
# python record_sim_episodes_optimized.py \
#     --task_name sim_stir_scripted \
#     --num_episodes 10
# python visualize_episodes.py \
#     --dataset_dir generated_data/sim_stir_scripted
python visualize_episodes.py \
    --dataset_dir evaluation_data/sim_stir_scripted
# conda install -c conda-forge glew
# $ conda install -c conda-forge mesalib
# $ conda install -c anaconda mesa-libgl-cos6-x86_64
# $ conda install -c menpo glfw3
# conda install -c anaconda libegl1-mesa libegl1-mesa-dev