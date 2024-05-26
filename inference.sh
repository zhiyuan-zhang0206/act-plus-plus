ckpt_path=${1:-"newest"}

original_dir=$(pwd)

cd /home/users/ghc/zzy/act-plus-plus
conda activate zzy-aloha-3
export CUDA_VISIBLE_DEVICES=2
# Pass the ckpt_path parameter to the script
python record_sim_episodes_with_model.py --num_episodes 10 --task_name stir --reward_threshold 2.9 --center_location --model_ckpt_path $ckpt_path
python record_sim_episodes_with_model.py --num_episodes 10 --task_name openlid --reward_threshold 2.1 --center_location --model_ckpt_path $ckpt_path
python record_sim_episodes_with_model.py --num_episodes 10 --task_name transfercube --reward_threshold 1.9 --center_location --model_ckpt_path $ckpt_path
conda deactivate

cd $original_dir
