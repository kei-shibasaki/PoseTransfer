DEVICE=2
CONFIG_FILE="config/config_market_drop_short.json"
LOGDIR_PRE="results/market_drop_short_pre"
LOGDIR_FINE="results/market_drop_short_fine"

CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_pre --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m train_codes.train_fine --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 -m pre --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation -m pre --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 -m fine --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation -m fine --config $CONFIG_FILE

echo >> $LOGDIR_PRE/results.txt
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR_PRE/GT $LOGDIR_PRE/generated --device cuda:0 >> $LOGDIR_PRE/results.txt

echo >> $LOGDIR_FINE/results.txt
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR_FINE/GT $LOGDIR_FINE/generated --device cuda:0 >> $LOGDIR_FINE/results.txt