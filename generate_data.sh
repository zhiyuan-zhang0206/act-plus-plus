cd /home/users/ghc/zzy/act-plus-plus/
# conda activate zzy-aloha-3
# python record_sim_episodes_optimized.py --task_name stir --num_episodes 50 --final_reward_threshold 3.0
# python visualize_episodes.py --dataset_dir generated_data/stir --save_image
# python visualize_episodes.py --dataset_dir generated_data/stir

# python record_sim_episodes_optimized.py --task_name openlid --num_episodes 50 --final_reward_threshold 2.5
# python visualize_episodes.py --dataset_dir generated_data/openlid --save_image
# python visualize_episodes.py --dataset_dir generated_data/openlid

# python record_sim_episodes_optimized.py --task_name transfercube --num_episodes 50 --final_reward_threshold 1.9
# python visualize_episodes.py --dataset_dir generated_data/transfercube --save_image
# python visualize_episodes.py --dataset_dir generated_data/transfercube

# conda activate zzy-rtx
# python convert_data_format.py --right_hand_relative False --absolute True --center_location True
# conda deactivate

cd /home/users/ghc/zzy/
source .bashrc
cd /home/users/ghc/zzy/act-plus-plus/dataset/bimanual_zzy/
conda activate zzy-rtx
set_proxy
tfds build
cd /home/users/ghc/zzy/act-plus-plus/
# python record_sim_episodes_optimized.py --task_name openlid --dataset_dir generated_data --num_episodes 500 --render_interval 10 --render_start 200
# python visualize_episodes.py --dataset_dir generated_data/openlid --save_image
# python visualize_episodes.py --dataset_dir generated_data/openlid




# cd /home/users/ghc/zzy/act-plus-plus/
# python record_sim_episodes_optimized.py --task_name stir --dataset_dir generated_data --num_episodes 50 --render_interval 1
# python visualize_episodes.py --dataset_dir generated_data/stir --save_image
# python visualize_episodes.py --dataset_dir generated_data/stir
# python record_sim_episodes_optimized.py --task_name openlid --dataset_dir generated_data --num_episodes 50 --render_interval 1
# python visualize_episodes.py --dataset_dir generated_data/openlid --save_image
# python visualize_episodes.py --dataset_dir generated_data/openlid

