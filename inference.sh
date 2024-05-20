original_dir=$(pwd)

cd /home/users/ghc/zzy/act-plus-plus
conda activate zzy-aloha-3
# python record_sim_episodes_with_model.py --num_episodes 10 --task_name stir --reward_threshold 2.9
python record_sim_episodes_with_model.py --num_episodes 10 --task_name openlid --reward_threshold 2.0
python record_sim_episodes_with_model.py --num_episodes 10 --task_name transfercube --reward_threshold 1.9
conda deactivate

cd $original_dir
