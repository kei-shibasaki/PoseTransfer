DEVICE=2
CONFIG_FILE="config/config_market_small.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_pre --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_fine --config $CONFIG_FILE