import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from globals import *


def generate_features(data: pd.DataFrame):
    """
        Generate additional features
    """
    weeks_per_year = 52.1429
    weeks = data['date'].map(lambda x: x.weekofyear)
    data['week_sin'] = np.sin(weeks * (2 * np.pi / weeks_per_year))
    data['week_cos'] = np.cos(weeks * (2 * np.pi / weeks_per_year))

    # apply one hot encoding to dpt_num_department
    data = pd.get_dummies(data, columns=['dpt_num_department'])
    return data


def transform_data(data, info):
    """
        Prepares the data to be ingested by an ML model
    """
    data = data.merge(info, on='but_num_business_unit', how='left')

    data['date'] = pd.to_datetime(data['day_id'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    # add new features
    data = generate_features(data)
    if MODEL_TYPE == 'lstm':
        train_x, train_y = create_timeseries_windows(data[FEATURES + [LABEL]],
                                                     input_wnd_size=INPUT_WEEKS,
                                                     output_wnd_size=OUTPUT_WEEKS)
    elif MODEL_TYPE == "linear":
        train_x = np.array(data[FEATURES])
        train_y = np.array(data[LABEL])
    else:
        raise ValueError(f'Model type {MODEL_TYPE} will be available soon.')
    return train_x, train_y


def create_timeseries_windows(data, input_wnd_size, output_wnd_size, label=LABEL):
    """
        Creates windows for training an RNN model

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
            for x, y in dataset.batch(1).prefetch(1).as_numpy_iterator():
                train_x.append(x)
                train_y.append(np.expand_dims(y, 1))

    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    return train_x, train_y
