import os
import glob
from utils.lmdb_util import make_lmdb_from_imgs

def create_lmdb():
    folder_path = 'dataset/fashion/train'
    lmdb_path = 'datasets/fashion/train_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'dataset/fashion/test'
    lmdb_path = 'datasets/fashion/test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'dataset/market/train'
    lmdb_path = 'datasets/market/train_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'dataset/market/test'
    lmdb_path = 'datasets/market/test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys(folder_path):

    print('Reading image path list ...')
    img_path_list = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    create_lmdb()

