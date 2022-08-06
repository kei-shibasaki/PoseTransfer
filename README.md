# Pose-aware Disentangled Multiscale Transformer for Pose Guided Person Image Generation

# Installation
The current code is tested with
- Debian GNU/Linux 10
- Intel(R) Xeon(R) CPU E5-1620 v4 @ 3.50GHz OR AMD EPYC 7232P 8-Core Processor
- NVIDIA GeForce GTX 1080 Ti OR NVIDIA GeForce RTX 3090

### Conda Installation
```python
# 1. Create a conda virtual environment.
conda create -n pose
conda activate pose
# 2. Install dependency
conda install pip python=3.9.9 -c conda-forge
conda install cudatoolkit=11.3 pytorch-gpu=1.10.0 torchvision=0.10.1 -c conda-forge
conda install requests matplotlib tqdm pandas easydict -c conda-forge
conda install scipy=1.7.3 scikit-image=0.19.1 -c conda-forge
conda install lpips=0.1.3 -c conda-forge
pip install opencv-python==4.5.5 pytorch-fid==0.2.1
```

## Download Dataset
The same dataset as [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention) is used for our work.
### Deep Fashion
- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00?resourcekey=0-fsjVShvqXP2517KnwaZ0zw) Benchmark.

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then put the obtained folder img_highres under the `./dataset/fashion directory`.

- Download train/test key points annotations and the dataset list from [Google Drive](https://drive.google.com/drive/folders/1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL) including `fashion-pairs-train.csv, fashion-pairs-test.csv, fashion-annotation-train.csv, fashion-annotation-train.csv, train.lst, test.lst`. Put these files under the `./dataset/fashion` directory.

Run the following code to split the train/test dataset.
```python
python script/generate_fashion_datasets.py
```

### Market-1501
- Download the Market-1501 dataset from [here](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ). Rename bounding_box_train and bounding_box_test as train and test, and put them under the `./dataset/market` directory
- Download train/test key points annotations from [Google Drive](https://drive.google.com/drive/folders/1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL) including `market-pairs-train.csv, market-pairs-test.csv, market-annotation-train.csv, market-annotation-train.csv`. Put these files under the `./dataset/market directory`.

# Evaluation
1. Edit `$DEVICE, $CONFIG_FILE, $LOG_DIR` in `evaluation.sh`. 
`$DEVICE` is the index of GPU for evaluation. 
`$CONFIG_FILE` is the path to the config JSON file.
`$LOG_DIR` is the path where logs are output.
2. Run `evaluation.sh`.
3. We use pytorch-fid to calculate FID score.
So, if you want to compute FID score, run the follow command.
```python
python -m pytorch_fid path/to/generated_image path/to/ground_truth
```

# Training
1. (Optional) If you set `enable_line_nortify: true` in JSON config file and
create `line_nortify_token.json` as `{"token": YOUR_TOKEN}`,
you can receive the progress of your training in your LINE Nortify by 50000 Steps.
2. Edit `$DEVICE, $CONFIG_FILE, $LOG_DIR` in `train.sh`. 
`$DEVICE` is the index of GPU for training. 
`$CONFIG_FILE` is the path to the config JSON file.
`$LOG_DIR` is the path where logs are output.
3. Run `train.sh`.

