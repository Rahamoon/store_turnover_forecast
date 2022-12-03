import os

import numpy as np
import pandas as pd
import tensorflow as tf

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', "data")
    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

    train_data = add_features(train_data)
    test_data = add_features(train_data)

    train_data['turnover'], train_mean, train_std = normalize_data(train_data['turnover'])
    test_data['turnover'], *_ = normalize_data(train_data['turnover'], train_mean, train_std)

    val_data = test_data.iloc[:int(0.2 * len(test_data))].copy()
    test_data = test_data.iloc[int(0.2 * len(test_data)):].copy()

    return [data[['dpt_num_department', 'but_num_business_unit', 'year', 'week_sin', 'week_cos', 'turnover']] for data
            in
            [train_data, val_data, test_data]]


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


def add_features(data: pd.DataFrame):
    """
        generate datetime features from the data
    Args:
        data:

    Returns:
        data with new features
    """
    data['date'] = pd.to_datetime(data['day_id'], format='%Y-%m-%d')
    data['week'] = data['date'].map(lambda x: x.weekofyear)
    data['year'] = data['date'].map(lambda x: x.year)

    weeks_per_year = 52.1429
    data['week_sin'] = np.sin(data['week'] * (2 * np.pi / weeks_per_year))
    data['week_cos'] = np.cos(data['week'] * (2 * np.pi / weeks_per_year))
    return data


