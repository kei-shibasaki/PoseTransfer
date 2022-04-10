DEVICE=3
LOGDIR="results/market_large_fine"

echo >> $LOGDIR/results.txt
CUDA_VISIBLE_DEVICES=$DEVICE python -m pytorch_fid $LOGDIR/GT $LOGDIR//generated --device cuda:0 >> $LOGDIR/results.txt