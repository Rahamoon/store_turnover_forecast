import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from globals import *


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


def transform_data(data, info, mode='timeseries'):
    """
        Prepare the data to be ingested by an ML model
    Args:
        data:
        info:

    Returns:
        data
    """
    data = data.merge(info, on='but_num_business_unit', how='left')

    data['date'] = pd.to_datetime(data['day_id'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    # add new features
    data = generate_features(data)
    if mode == 'timeseries':
        train_x, train_y = create_timeseries_windows(data[FEATURES + [LABEL]],
                                                     input_wnd_size=INPUT_WEEKS,
                                                     output_wnd_size=OUTPUT_WEEKS)
    else:
        train_x = np.array(data[FEATURES])
        train_y = np.array(data[LABEL])
    return train_x, train_y


def create_timeseries_windows(data, input_wnd_size, output_wnd_size, label=LABEL):
    """

    Args:
        data:
        input_wnd_size:
        output_wnd_size:
        label:

    Returns:

    """

    train_x = []
    train_y = []
    window_size = input_wnd_size + output_wnd_size
    label_col_idx = data.columns.get_loc(label)
    for store in tqdm(data['but_num_business_unit'].unique()):
        data_store = data.loc[data['but_num_business_unit'] == store]
        for dpt_col in [col for col in data_store.columns if col.startswith('dpt_num_department')]:
            data_store_dpt = data_store.loc[data_store[dpt_col] == 1]
            if len(data_store_dpt) < window_size:
                continue
            dataset = tf.data.Dataset.from_tensor_slices(np.array(data_store_dpt))
            dataset = dataset.window(window_size, shift=1, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(window_size))
            dataset = dataset.map(lambda window: (window[:-output_wnd_size], window[-output_wnd_size:, label_col_idx]))
            for x, y in dataset:
                train_x.append(x.numpy())
                train_y.append(y.numpy().T)

    train_x = np.asarray(train_x)
    train_y = np.expand_dims(np.asarray(train_y), 2)
    return train_x, train_y
