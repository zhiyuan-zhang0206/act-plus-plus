original_dir=$(pwd)

cd /home/users/ghc/zzy/act-plus-plus
conda activate zzy-aloha-3
python record_sim_episodes_with_model.py --num_episodes 50 --task_name stir --reward_threshold 2.9 --center_location
python record_sim_episodes_with_model.py --num_episodes 50 --task_name openlid --reward_threshold 2.1 --center_location
python record_sim_episodes_with_model.py --num_episodes 50 --task_name transfercube --reward_threshold 2.1 --center_location
conda deactivate

cd $original_dir
