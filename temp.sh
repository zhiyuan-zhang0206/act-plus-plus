export CUDA_VISIBLE_DEVICES=2
. ./generate_data.sh
python convert_data_ACT.py
. ./train.sh
