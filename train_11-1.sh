DEVICE=1
CONFIG_FILE="config/config_fashion_large_onestep.json"
# LOGDIR_PRE="results/fashion_large_long_pre"
LOGDIR_FINE="results/fashion_large_onestep"

# CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_pre --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_fine --config $CONFIG_FILE

# CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 -m pre --config $CONFIG_FILE
# CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation -m pre --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 -m fine --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation -m fine --config $CONFIG_FILE

# echo >> $LOGDIR_PRE/results.txt
# CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR_PRE/GT $LOGDIR_PRE/generated --device cuda:0 >> $LOGDIR_PRE/results.txt

echo >> $LOGDIR_FINE/results.txt
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR_FINE/GT $LOGDIR_FINE/generated --device cuda:0 >> $LOGDIR_FINE/results.txt