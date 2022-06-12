DEVICE=0
CONFIG_FILE="experiments/fashion/config_fashion.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 16 --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.visualize_losses -a 0.9 -c $CONFIG_FILE

# CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR/GT $LOGDIR/generated --device cuda:0 >> $LOGDIR/results.txt
