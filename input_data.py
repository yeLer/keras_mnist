"""
Reference implementation of MNIST classification experiments.
"""

import numpy as np
import os
from scipy.ndimage import imread
from PIL import Image
from tqdm import tqdm


def get_mnist_data_iters(data_dir, train_size, test_size, full_test_set, seed):
    """
    Load MNIST data.

    Assumed data directory structure:
        training/
            0/
            1/
            2/
            ...
        testing/
            0/
            ...
    Parameters
    ----------
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    train_data, test_data : [(numpy.ndarray, str)]
        Each item reps a data sample (2-tuple of image and label)
        Images are numpy.uint8 type [0,255]
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class):
        loaded_data = []
        for category in tqdm(sorted(os.listdir(image_dir))):
            cat_path = os.path.join(image_dir, category)
            # 如果当前文件不是文件夹，或者以.开头（避免mac系统上的.DS_store文件），中止本次循环
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            if num_per_class is None:
                samples = sorted(os.listdir(cat_path))
            else:
                samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                img = imread(filepath, 'L')
                loaded_data.append((img, category))
        return loaded_data
    np.random.seed(seed)
    train_set = _load_data(os.path.join(data_dir, 'training'),
                           num_per_class=train_size // 10)  # need to change according to datasets
    test_path = os.path.join(data_dir, 'testing')
    test_set = _load_data(test_path,
                          num_per_class=None if full_test_set else test_size // 10)  # need to change according to datasets
    print("Test data:", test_path)
    return train_set, test_set

# this function is used to testing
if __name__ =="__main__":
    train_set, test_set = get_mnist_data_iters(data_dir='/home/yeler082/datasets/MNIST/',
                                               train_size=10000, test_size=2000, full_test_set=False, seed=20)
    x0_train_set, y0_test_set = train_set[6666][0], train_set[6666][1]
    image = Image.fromarray(x0_train_set)
    image.show()
    print(y0_test_set)

