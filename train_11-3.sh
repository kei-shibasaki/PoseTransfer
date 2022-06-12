DEVICE=0
CONFIG_FILE="config/config_market_arg_long.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_dev --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 32 --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.visualize_losses -a 0.9 -c $CONFIG_FILE
