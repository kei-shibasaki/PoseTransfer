DEVICE=3
CONFIG_FILE="config/config_market_small.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_l1 --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train --config $CONFIG_FILE