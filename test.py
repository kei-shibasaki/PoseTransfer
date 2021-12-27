from options.test_options import TestOptions
import data as Dataset
from util import visualizer
from itertools import islice
import numpy as np
import torch

if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)


