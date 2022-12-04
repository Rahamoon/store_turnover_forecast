import logging

import tensorflow as tf
from fire import Fire

from load import *
from model import build_model, compile_fit_model, plot_metrics

logger = logging.getLogger(__name__)


def train_linear(model_save_path=None, test=True):
    data = load_data()

    data = generate_features(data)

    train_data, val_data, test_data = split_data(data)

    features = ['dpt_num_department_73', 'dpt_num_department_88',
                'dpt_num_department_117', 'dpt_num_department_127',
                'but_num_business_unit', 'week_sin', 'week_cos',
                'but_latitude', 'but_longitude']
    label_col = 'turnover'

    # normalization layer
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(train_data[features])

    linear_model = build_model(output_size=1, normalization_layer=norm_layer)

    history = compile_fit_model(linear_model, train_data[features + [label_col]], val_data[features + [label_col]],
                                label_col=label_col,
                                batch_size=32,
                                epochs=25,
                                model_save_path=model_save_path)

    if test:
        performance = linear_model.evaluate(x=np.array(test_data[features], dtype=np.float),
                                            y=np.array(test_data[label_col]), verbose=2)
        logger.warning(f'Model performance on test data: "test_mean_squared_error"={performance[0]}')
    plot_metrics(history, os.path.join(model_save_path, 'model_training_metrics.png'))


if __name__ == "__main__":
    Fire(train_linear)
