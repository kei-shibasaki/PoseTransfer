<<<<<<< Updated upstream
DEVICE=3
MODEL_NAME="fashion_256x128_large_mod_onestep"
CONFIG_FILE="config/config_$MODEL_NAME.json"
LOGDIR_PRE="results/"$MODEL_NAME
LOGDIR_FINE="results/"$MODEL_NAME

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
=======
DEVICE=0
CONFIG_FILE="config/config_market_arg_long.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_dev --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 32 --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.visualize_losses -a 0.9 -c $CONFIG_FILE
>>>>>>> Stashed changes
