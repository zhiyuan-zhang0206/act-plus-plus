# table texture test



process_files() {
    local target_dir=$1
    local gen_number=$2
    for file in "$target_dir"/*_gen${gen_number}.xml; 
    do
        base_name=$(basename "$file" "_gen${gen_number}.xml")
        non_gen_file="$target_dir/$base_name.xml"
        if [ -f "$non_gen_file" ]; then
            echo "Deleting $non_gen_file"
            rm "$non_gen_file"
        fi
        new_xml_file="$target_dir/$base_name.xml"
        echo "Copying $file to $new_xml_file"
        cp "$file" "$new_xml_file"
    done
}
export CUDA_VISIBLE_DEVICES=2
TARGET_DIR="/home/users/ghc/zzy/act-plus-plus/assets"
ckpt="/home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-23_17-49-00"



# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen1.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# . /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" # 0 0 0
# conda activate zzy-aloha-train
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 3
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 9
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 2
# conda deactivate

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen2.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# . /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" # 0 7 0
# conda activate zzy-aloha-train
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 3
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 6
# conda deactivate

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen3.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# . /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" #  0 3 0
# conda activate zzy-aloha-train
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 1
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 3
# conda deactivate

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen0.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png

# rm /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# cp /home/users/ghc/zzy/act-plus-plus/assets/table_texture_gen0.png /home/users/ghc/zzy/act-plus-plus/assets/table_texture.png
# process_files "$TARGET_DIR" 0

conda activate zzy-aloha-train
process_files "$TARGET_DIR" 0
. /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt"
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 1
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 4

process_files "$TARGET_DIR" 1
. /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" # 2 8 1  0.43 0.90 0.55
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 1
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 4

process_files "$TARGET_DIR" 2
. /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" # 3 8 7  0.47 0.90 0.80
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 1
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 2

process_files "$TARGET_DIR" 3
. /home/users/ghc/zzy/act-plus-plus/inference.sh "$ckpt" # 0 9 0 0.37 0.97 0.45
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval                  # 0
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval            # 10
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval  # 2
# conda deactivate
conda activate zzy-aloha-3
cd /home/users/ghc/zzy/act-plus-plus
python visualize_episodes.py --dataset_dir ./evaluation_data
cd /home/users/ghc/zzy/open_x_embodiment-main
conda deactivate