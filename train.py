import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from globals import *

logger = logging.getLogger(__name__)


def build_linear_model(output_size, normalization_layer):
    return tf.keras.Sequential([
        normalization_layer,
        tf.keras.layers.Dense(units=output_size)
    ])


def build_lstm_model(output_size):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(units=output_size)
    ])


def compile_fit_model(model, train_x, train_y,
                      patience=2, batch_size=32, epochs=10, lr=0.1,
                      model_save_path=None):
    """
        Compile and train the model
    Args:
        model: tensorflow model
        train_x: train data set features
        train_y: train data set label
        patience: number of epochs of steady val_loss for triggering training stop
        batch_size: number of samples per batch
        epochs: number of epochs
        model_save_path: path to save the model file

    Returns:
        history: model training logs
    """
    logger.warning(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
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

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(x=train_x,
                        y=train_y,
                        epochs=epochs,
                        validation_split=0.3,
                        callbacks=[early_stopping, model_checkpoint_callback],
                        batch_size=batch_size,
                        verbose=2)
    return history


def train_model(train_x, train_y, model_type="lstm", model_save_path=None):
    """

    Args:
        train_x: train input in numpy array format
        train_y: train label in numpy array format
        model_type: "lstm" | "linear
        model_save_path: path to save the model

    Returns:

    """

    if model_type == 'linear':
        # normalization layer
        norm_layer = tf.keras.layers.Normalization(axis=-1)
        norm_layer.adapt(train_x)
        # train
        model = build_linear_model(output_size=1, normalization_layer=norm_layer)
    elif model_type == 'lstm':
        model = build_lstm_model(output_size=OUTPUT_WEEKS)

    history = compile_fit_model(model,
                                train_x, train_y,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                lr=LEARNING_RATE,
                                model_save_path=model_save_path)

    plot_metrics(history, os.path.join(model_save_path, 'model_training_metrics.png'))
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
