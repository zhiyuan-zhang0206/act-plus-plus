ckpt_path="/home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-23_17-49-00"

original_dir=$(pwd)

cd /home/users/ghc/zzy/act-plus-plus
conda activate zzy-aloha-3
export CUDA_VISIBLE_DEVICES=3
python record_sim_episodes_with_model.py --num_episodes 10 --task_name stir --reward_threshold 2.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction          "insert the spoon into the cup and mix" # 0
python record_sim_episodes_with_model.py --num_episodes 10 --task_name stir --reward_threshold 2.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction          "place the spoon in the cup and swirl it around" # 0
python record_sim_episodes_with_model.py --num_episodes 10 --task_name stir --reward_threshold 2.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction          "dip the spoon into the cup and stir the contents" # 0

python record_sim_episodes_with_model.py --num_episodes 10 --task_name openlid --reward_threshold 2.1 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction       "lift the lid off the cup" # 0
python record_sim_episodes_with_model.py --num_episodes 10 --task_name openlid --reward_threshold 2.1 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction       "remove the cup's lid" # 0
python record_sim_episodes_with_model.py --num_episodes 10 --task_name openlid --reward_threshold 2.1 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction       "take the lid off the cup" # 0

python record_sim_episodes_with_model.py --num_episodes 10 --task_name transfercube --reward_threshold 1.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction  "move the cube from your left hand to your right hand" # 1
python record_sim_episodes_with_model.py --num_episodes 10 --task_name transfercube --reward_threshold 1.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction  "pass the cube from the left hand to the right hand" # 4
python record_sim_episodes_with_model.py --num_episodes 10 --task_name transfercube --reward_threshold 1.9 --center_location --model_ckpt_path $ckpt_path --overwrite_language_instruction  "shift the cube from the left hand to the right hand" # 0
conda deactivate

cd $original_dir
