DEVICE="0"
CONFIG_FILE="config/config_fashion.json"
RES_DIR="results/fashion/"

GEN_DIR=$RES_DIR"generated"
GT_DIR=$RES_DIR"GT"
LOG_PATH=$RES_DIR'results.txt'

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.generate_images -b 32 --config $CONFIG_FILE
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.evaluation --config $CONFIG_FILE

CUDA_VISIBLE_DEVICES=$DEVICE python -m eval_codes.visualize_losses -a 0.9 -c $CONFIG_FILE
echo >> $LOG_PATH
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $GEN_DIR $GT_DIR >> $LOG_PATH