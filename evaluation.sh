DEVICE=3
CONFIG_FILE="config/config_fashion_large.json"
MODE="pre"
LOGDIR="results/fashion_large_pre"

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 -m $MODE --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation -m $MODE --config $CONFIG_FILE

echo >> $LOGDIR/results.txt
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR/GT $LOGDIR/generated --device cuda:0 >> $LOGDIR/results.txt