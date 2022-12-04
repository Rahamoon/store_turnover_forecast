import os

import numpy as np
import pandas as pd


def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '', "data")
    data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    info = pd.read_csv(os.path.join(data_path, 'bu_feat.csv'))
    data = data.merge(info, on='but_num_business_unit', how='left')
    data['date'] = pd.to_datetime(data['day_id'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    return data


def split_data(data):
    train_data = data.iloc[:int(0.6 * len(data))].copy()
    val_data = data.iloc[int(0.6 * len(data)):int(0.8 * len(data))].copy()
    test_data = data.iloc[int(0.8 * len(data)):].copy()
    return train_data, val_data, test_data


def normalize_data(turnover: pd.Series, data_mean=None, data_std=None):
    """
        Normalize the turnover values
    Args:
        turnover: turnover values
        data_mean: mean of the turnover if precomputed
        data_std: standard deviation of the turnover if precomputed

    Returns:
        dataframe with normalized turnover
    """
    data = turnover.copy(deep=True)
    if data_mean is None:
        data_mean = data.mean()
    if data_std is None:
        data_std = data.std()
    data = (data - data_mean) / data_std
    return data, data_mean, data_std


def generate_features(data: pd.DataFrame):
    """
        generate datetime features from the data
    Args:
        data:

    Returns:
        data with new features
    """
    weeks_per_year = 52.1429
    weeks = data['date'].map(lambda x: x.weekofyear)
    data['week_sin'] = np.sin(weeks * (2 * np.pi / weeks_per_year))
    data['week_cos'] = np.cos(weeks * (2 * np.pi / weeks_per_year))

    # apply one hot encoding to dpt_num_department
    data = pd.get_dummies(data, columns=['dpt_num_department'])
    return data
