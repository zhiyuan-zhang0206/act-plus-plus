# sleep 3600
cd /home/users/ghc/zzy/act-plus-plus
# python convert_data_ACT.py
conda activate zzy-aloha-train
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt_hard --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --hard_mode
# WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt_hard --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --hard_mode
# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt_hard --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --hard_mode

# WANDB_MODE=offline python3 imitate_episodes.py --task_name transfercube-act --ckpt_dir transfercube_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval
# WANDB_MODE=offline python3 imitate_episodes.py --task_name stir-act --ckpt_dir stir_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0 --eval
WANDB_MODE=offline python3 imitate_episodes.py --task_name openlid-act --ckpt_dir openlid_ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 2000  --lr 1e-5 --seed 0

conda deactivate