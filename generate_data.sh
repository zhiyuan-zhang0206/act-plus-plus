cd /home/users/ghc/zzy/act-plus-plus/
python record_sim_episodes_optimized.py --task_name stir --dataset_dir generated_data --num_episodes 5 --render_interval 10
python visualize_episodes.py --dataset_dir generated_data/stir --save_image
python visualize_episodes.py --dataset_dir generated_data/stir
python record_sim_episodes_optimized.py --task_name openlid --dataset_dir generated_data --num_episodes 5 --render_interval 10
python visualize_episodes.py --dataset_dir generated_data/openlid --save_image
python visualize_episodes.py --dataset_dir generated_data/openlid




# cd /home/users/ghc/zzy/act-plus-plus/
# python record_sim_episodes_optimized.py --task_name stir --dataset_dir generated_data --num_episodes 50 --render_interval 1
# python visualize_episodes.py --dataset_dir generated_data/stir --save_image
# python visualize_episodes.py --dataset_dir generated_data/stir
# python record_sim_episodes_optimized.py --task_name openlid --dataset_dir generated_data --num_episodes 50 --render_interval 1
# python visualize_episodes.py --dataset_dir generated_data/openlid --save_image
# python visualize_episodes.py --dataset_dir generated_data/openlid

