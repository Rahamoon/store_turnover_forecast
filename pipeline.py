import logging
import os.path

import numpy as np
import tensorflow as tf
from fire import Fire

from load import load_data
from train import train_model
from transform import transform_data
from globals import MODEL_TYPE, FEATURES

logger = logging.getLogger(__name__)


def model_train_pipeline(model_save_path=None):
    """

    Args:
        model_save_path: path to save the trained model

    """
    logger.warning("loading the data..")
    data, info = load_data('train')

    logger.warning("transforming the data..")
    train_x, train_y = transform_data(data, info)

    logger.warning("training..")
    history, test_perf = train_model(train_x, train_y, model_type=MODEL_TYPE, model_save_path=model_save_path)


def model_predict(model_path, result_save_path=None):
    """
        Load model from model_path, perform prediction and save results to result_save_path
    Args:
        model_path: path to the model artifact
        result_save_path: path to save prediction results

    """
    data, info = load_data('test')
    train_x, train_y = transform_data(data, info)
    model = tf.keras.models.load_model(model_path)
    data['predicted_turnover'] = model.predict(x=np.array(data[FEATURES])).flatten()
    if result_save_path is not None:
        data.to_csv(os.path.join(result_save_path, 'predictions.csv'), index_col=0)
    return data


if __name__ == "__main__":
    Fire(model_train_pipeline)
