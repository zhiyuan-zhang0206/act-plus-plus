# sleep 5400

# python record_sim_episodes_with_model.py --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-17_18-45-13 --always_refresh True --absolute True --num_episodes 10 --task_name stir --reward_threshold 2.9 > output.txt
# python visualize_episodes.py --dataset_dir ./evaluation_data/stir/0

python record_sim_episodes_with_model.py --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-17_18-45-13 --always_refresh True --absolute True --num_episodes 3 --task_name openlid --reward_threshold 2.8 >> output.txt
python visualize_episodes.py --dataset_dir ./evaluation_data/openlid/0

# python record_sim_episodes_with_model.py --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-17_18-45-13 --always_refresh True --absolute True --num_episodes 10 --task_name transfercube --reward_threshold 1.9 >> output.tx
# python visualize_episodes.py --dataset_dir ./evaluation_data/transfercube/0/

# python record_sim_episodes_with_model.py --model_ckpt_path /home/users/ghc/zzy/open_x_embodiment-main/rt_1_x_jax_bimanual/2024-05-17_18-45-13 --always_refresh True --absolute True --num_episodes 1 --task_name transfercube