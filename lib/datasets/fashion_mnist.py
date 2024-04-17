"""
Fashion MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
"""

from os import makedirs
from os.path import exists, join
import gzip

from sklearn.datasets._base import RemoteFileMetadata, _fetch_remote
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch

import numpy as np
import logging


logger = logging.getLogger(__name__)

ARCHIVES = [
    RemoteFileMetadata(
        filename='train-images-idx3-ubyte.gz',
        url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        checksum=('3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84')),
    RemoteFileMetadata(
        filename='train-labels-idx1-ubyte.gz',
        url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        checksum=('a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845')),
    RemoteFileMetadata(
        filename='test-images-idx3-ubyte.gz',
        url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        checksum=('346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073')),
    RemoteFileMetadata(
        filename='test-labels-idx1-ubyte.gz',
        url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        checksum=('67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5'))
]


def fetch_fashion_mnist(data_home=None, download_if_missing=True, subset='all', return_X_y=False):
    """
    Load the Fashion MNIST dataset (classification).

    Note: Based on https://github.com/scikit-learn/scikit-learn
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    for archive in ARCHIVES:
        filepath = join(data_home, archive.filename)
        if not exists(filepath):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            logger.info('Downloading Fashion mnist from {} to {}'.format(
                archive.url, filepath))
            archive_path = _fetch_remote(archive, dirname=data_home)

    if return_X_y:

        DESCR = '''
            Fashion-MNIST is a dataset of Zalando's article images—consisting of 
            a training set of 60,000 examples and a test set of 10,000 examples. 
            Each example is a 28x28 grayscale image, associated with a label from 
            10 classes. We intend Fashion-MNIST to serve as a direct drop-in 
            replacement for the original MNIST dataset for benchmarking machine 
            learning algorithms. It shares the same image size and structure of 
            training and testing splits.
            '''

        feature_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        if subset == 'train':
            X, y = _load_X_y(data_home, 'train')
            return Bunch(
                data=X,
                target=y,
                feature_names=feature_names,
                DESCR=DESCR
            )
        elif subset == 'test':
            X, y = _load_X_y(data_home, 'test')
            return Bunch(
                data=X,
                target=y,
                feature_names=feature_names,
                DESCR=DESCR
            )
        X_train, y_train = _load_X_y(data_home, 'train')
        X_test, y_test = _load_X_y(data_home, 'test')
        return Bunch(
            data={'train': X_train, 'test': X_test},
            target={'train': y_train, 'test': y_test},
            feature_names=feature_names,
            DESCR=DESCR
        )


def _load_X_y(path, subset='train'):
    """
    Load MNIST data from `path`.

    Note: Based on
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    y_path = join(path, '{}-labels-idx1-ubyte.gz'.format(subset))
    X_path = join(path, '{}-images-idx3-ubyte.gz'.format(subset))

    with gzip.open(y_path, 'rb') as y_file:
        y = np.frombuffer(y_file.read(), dtype=np.uint8,
                          offset=8)
    with gzip.open(X_path, 'rb') as X_file:
        X = np.frombuffer(X_file.read(), dtype=np.uint8,
                          offset=16).reshape(len(y), 784)
    return X, y
