"""
20 Languages Dataset:
https://huggingface.co/datasets/papluca/language-identification
"""

from os import makedirs
from os.path import exists, join

from sklearn.datasets.base import RemoteFileMetadata, _fetch_remote
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch

import pandas as pd
import logging


logger = logging.getLogger(__name__)

ARCHIVES = [
    RemoteFileMetadata(
        filename='languages_train.csv',
        url='https://huggingface.co/datasets/papluca/language-identification/resolve/main/train.csv',
        checksum=('f180d78a1f0e758fd33bb1bae37f62eebc538d78ece2affb3d05a967850ba474')),
    RemoteFileMetadata(
        filename='languages_test.csv',
        url='https://huggingface.co/datasets/papluca/language-identification/resolve/main/test.csv',
        checksum=('cb7dfe272142815573b735b5d555d42d28d0d648187020f2d2eb3eebd772e759'))

]


def fetch_20languages(data_home=None, download_if_missing=True, subset='all', return_X_y=False):
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    for archive in ARCHIVES:
        filepath = join(data_home, archive.filename)
        if not exists(filepath):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            logger.info('Downloading Languages from {} to {}'.format(
                archive.url, filepath))
            archive_path = _fetch_remote(archive, dirname=data_home)
    if return_X_y:
        DESCR = (
            '20 Languages Dataset\n'
            '--------------------\n'
            'The Language Identification dataset is a collection of 90k samples consisting of text passages and corresponding language label. This dataset was created by collecting data from 3 sources: [Multilingual Amazon Reviews Corpus](https://huggingface.co/datasets/amazon_reviews_multi), [XNLI](https://huggingface.co/datasets/xnli), and [STSb Multi MT](https://huggingface.co/datasets/stsb_multi_mt).\n'
            '\n'
            'The Language Identification dataset contains text in 20 languages, which are:\n'
            'arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)\n'
            '\n'
            'For each instance, there is a string for the text and a string for the label (the language tag). Here is an example:\n'
            "{'labels': 'fr', 'text': 'Conforme à la description, produit pratique.'}"
        )
        if subset == 'train':
            train_df = _load_X_y(data_home, 'train')
            return Bunch(
                data=train_df,
                DESCR=DESCR
            )
        elif subset == 'test':
            test_df = _load_X_y(data_home, 'test')
            return Bunch(
                data=test_df,
                DESCR=DESCR
            )
        train_df = _load_X_y(data_home, 'train')
        test_df = _load_X_y(data_home, 'test')
        return Bunch(
            data={'train': train_df, 'test': test_df},
            DESCR=DESCR
        )


def _load_X_y(path, subset='train'):
    return pd.read_csv(join(path, 'languages_{}.csv'.format(subset)))
