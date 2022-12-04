import logging
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def build_model(output_size, normalization_layer):
    """
        build a sequential linear regression model
    Args:
        output_size: size of the output label
        normalization_layer: normalization layer adapted to the training set

    Returns:
        linear model
    """
    return tf.keras.Sequential([
        normalization_layer,
        tf.keras.layers.Dense(units=output_size)
    ])


def compile_fit_model(model, train, val, label_col='turnover', patience=2, batch_size=32, epochs=10, lr=0.1, model_save_path=None):
    """
        Compile and train the model
    Args:
        model: tensorflow model
        train: train data set
        val: validation data set
        label_col: label column to predict
        patience: number of epochs of steady val_loss for triggering training stop
        batch_size: number of samples per batch
        epochs: number of epochs
        model_save_path: path to save the model file

    Returns:
        history: model training logs
    """
    if model_save_path is None:
        model_save_path = os.path.join(os.path.dirname(__file__), 'models')
    logger.info(f"Model will be saved in {model_save_path}")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(x=np.array(train.drop(labels=label_col, axis=1)),
                        y=np.array(train[label_col]),
                        epochs=epochs,
                        validation_data=(
                            np.array(val.drop(labels=label_col, axis=1)),
                            np.array(val[label_col])
                        ),
                        callbacks=[early_stopping, model_checkpoint_callback],
                        batch_size=batch_size,
                        verbose=2)
    return history


def plot_metrics(history, save_path=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save_path is not None:
        plt.savefig(save_path)