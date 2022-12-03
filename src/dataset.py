import numpy as np
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, input_width, label_width, shift, train_data, val_data, test_data, label_columns=None):
        """
            This class generates tf.data.Dataset for training testing and validation by creating
             batches of windows splitted into (features, labels) pairs

             It was copied and adapted from
                https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

        Args:
            input_width: number of time steps of the input window
            label_width: number of time steps of the label window
            shift: number of time steps in the future to be predicted
            train_data: train dataframe
            val_data: validation dataframe
            test_data: testing dataframe
            label_columns: label columns to predict , example= ['turnover']
        """
        # Store the raw data.
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # Work out the label column indices.
        self.label_columns = ['turnover']
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_data.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def make_dataset(self, data):
        datasets = []
        # generate time series vectors at store-department level
        for store in data['but_num_business_unit'].unique():
            for dpt in data.loc[data['but_num_business_unit'] == store]['dpt_num_department'].unique():
                data_store = data.query("but_num_business_unit==@store and dpt_num_department==@dpt").reset_index(
                    drop=True)
                data_store = np.array(data_store, dtype=np.float32)

                ds = tf.keras.utils.timeseries_dataset_from_array(
                    data=data_store,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=1,
                    shuffle=True,
                    batch_size=32)
                datasets.append(ds)

        ds = datasets[0]
        for i in range(1, len(datasets)):
            ds = ds.concatenate(datasets[i])

        ds = ds.map(self.split_window)
        return ds

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    @property
    def train(self):
        return self.make_dataset(self.train_data)

    @property
    def val(self):
        return self.make_dataset(self.val_data)

    @property
    def test(self):
        return self.make_dataset(self.test_data)


