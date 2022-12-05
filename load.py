import os

import pandas as pd


def load_data(dataset='train'):
    data_path = os.path.join(os.path.dirname(__file__), "data")

    if dataset == 'train':
        data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    elif dataset == 'test':
        data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    else:
        raise ValueError(f'Unknown dataset : {dataset}')

    info = pd.read_csv(os.path.join(data_path, 'bu_feat.csv'))

    return data, info
