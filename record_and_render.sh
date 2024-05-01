# python record_sim_episodes_optimized.py \
#     --task_name stir \
#     --num_episodes 50
# python visualize_episodes.py \
#     --dataset_dir generated_data/stir
python record_sim_episodes_with_model.py \
    --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-23_15-06-01/checkpoint_300 \
    --task_name stir \
    --num_episodes 5 \
    --frame_interval 30 \
    --always_refresh \
    --start_index 0
    # --dropout_train \
python visualize_episodes.py \
    --dataset_dir evaluation_data/stir/0

python record_sim_episodes_with_model.py \
    --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-23_15-17-56/checkpoint_600 \
    --task_name stir \
    --num_episodes 5 \
    --frame_interval 30 \
    --always_refresh \
    --start_index 1
    # --dropout_train \
python visualize_episodes.py \
    --dataset_dir evaluation_data/stir/1


python record_sim_episodes_with_model.py \
    --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-24_09-57-24/checkpoint_500 \
    --task_name stir \
    --num_episodes 5 \
    --frame_interval 30 \
    --always_refresh \
    --start_index 2
    # --dropout_train \
python visualize_episodes.py \
    --dataset_dir evaluation_data/stir/2


python record_sim_episodes_with_model.py \
    --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-24_10-18-31/checkpoint_500 \
    --task_name stir \
    --num_episodes 5 \
    --frame_interval 30 \
    --always_refresh \
    --start_index 3
    # --dropout_train \
python visualize_episodes.py \
    --dataset_dir evaluation_data/stir/3


python record_sim_episodes_with_model.py \
    --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-04-24_10-40-36/checkpoint_650 \
    --task_name stir \
    --num_episodes 5 \
    --frame_interval 30 \
    --always_refresh \
    --start_index 4
    # --dropout_train \
python visualize_episodes.py \
    --dataset_dir evaluation_data/stir/4




# conda install -c conda-forge glew
# conda install -c conda-forge mesalib
# conda install -c anaconda mesa-libgl-cos6-x86_64
# conda install -c menpo glfw3
# conda install -c anaconda libegl1-mesa libegl1-mesa-dev