from functools import partial as _partial

import numpy as np
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    """Generate windowed datasets.

    Parameters
    ----------
    train_df: pandas.DataFrame or list of DataFrame
        The training data.

    val_df: pandas.DataFrame or list of DataFrame
        The validation data.

    test_df: pandas.DataFrame or list of DataFrame
        The test data.

    x_width: int
        The width of the inputs.

    y_width: int
        The width of the outputs.

    x_columns: list of str
        The input features.

    y_columns: list of str
        The output features.

    sequence_stride: int, optional, default=1
        Period between successive output sequences.

    offset: int
        If positive, make predictions `offset` time steps into the future.
        If negative, start from the past.

    batch_size: int, optional, default=128
        Number of timeseries samples in each batch of the training set (except maybe the last one).
        For the validation and test sets, the actual batch size used is 2 * batch_size.

    shuffle: boolean, optional, default=True
        Whether to shuffle output samples, or instead draw them in chronological order.

    seed: int, optional, default=None
        Random seed for shuffling.
    """

    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        x_width,
        y_width,
        x_columns,
        y_columns,
        sequence_stride=1,
        offset=0,
        batch_size=128,
        shuffle=True,
        seed=None,
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.sequence_stride = sequence_stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.x_columns = x_columns
        self.x_columns_indices = {name: i for i, name in enumerate(x_columns)}
        self.y_columns = y_columns
        self.y_columns_indices = {name: i for i, name in enumerate(y_columns)}

        if isinstance(train_df, (tuple, list)):
            self.column_indices = {
                name: i for i, name in enumerate(train_df[0].columns)
            }
        else:
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.x_width = x_width
        self.y_width = y_width
        self.offset = offset
        self.total_window_size = max(x_width, x_width + offset)

        self.x_slice = slice(0, x_width)
        self.x_indices = np.arange(self.total_window_size)[self.x_slice]

        y_start = x_width + offset - y_width
        assert y_start >= 0, "The start index of y is smaller 0."
        self.y_slice = slice(y_start, y_start + y_width)
        self.y_indices = np.arange(self.total_window_size)[self.y_slice]

    def split_window(self, window):
        """
        Given a window, the split_window method will convert it to a window
        of inputs and a window of outputs.
        """
        # Dimensions are [batch_size, time_steps, features]
        inputs = window[:, self.x_slice, :]
        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.x_columns],
            axis=-1,
        )

        outputs = window[:, self.y_slice, :]
        outputs = tf.stack(
            [outputs[:, :, self.column_indices[name]] for name in self.y_columns],
            axis=-1,
        )
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.x_width, None])
        outputs.set_shape([None, self.y_width, None])
        return inputs, outputs

    def make_dataset(self, data, batch_size, shuffle=True, concat=False):
        """
        Slice the data to create batches of sequences.

        Parameters
        ----------
        data: pandas.DataFrame or list of DataFrame
            The data to be sliced.

        batch_size: int
            Number of timeseries samples in each batch (except maybe the last one).

        shuffle: boolean, optional, default=True
            Whether to shuffle output samples, or instead draw them in chronological order.

        concat: boolean, optional, default=False
            Only has an effect when `data` is a list of DataFrame. If True, use the method
            `Dataset.concatenate` to combine the data. Useful when plotting the predictions.
            If False, use `Dataset.sample_from_datasets` to interleave the data. It's better
            to set it False when training neural networks.
        """
        slicer = _partial(
            keras.preprocessing.timeseries_dataset_from_array,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=shuffle,
            batch_size=None,
        )
        if isinstance(data, (tuple, list)):
            lst = [slicer(data=d) for d in data]
            if concat:
                for i, ds in enumerate(lst):
                    if i == 0:
                        dataset = ds
                    else:
                        dataset = dataset.concatenate(ds)
            else:
                dataset = tf.data.Dataset.sample_from_datasets(lst, seed=self.seed)
        else:
            dataset = slicer(data=data)
        return dataset.batch(batch_size).map(self.split_window)

    def train_set(self, shuffle=None, concat=False):
        """
        Create windowed training set containing batches of sequences.

        Parameters
        ----------
        shuffle: boolean, optional, default=None
            Whether to shuffle output samples, or instead draw them in chronological order.
            If None, use `self.shuffle` defined during initialization.

        concat: boolean, optional, default=False
            Only has an effect when `self.train_df` is a list of DataFrame. If True, use the 
            method `Dataset.concatenate` to combine the data. Useful when plotting the 
            predictions. If False, use `Dataset.sample_from_datasets` to interleave the data. 
            It's better to set it False when training neural networks.
        """
        if shuffle is None:
            shuffle = self.shuffle
        return self.make_dataset(self.train_df, self.batch_size, shuffle, concat)

    def val_set(self, shuffle=None, concat=False):
        """
        Create validation set containing batches of sequences.

        Parameters
        ----------
        shuffle: boolean, optional, default=None
            Whether to shuffle output samples, or instead draw them in chronological order.
            If None, use `self.shuffle` defined during initialization.

        concat: boolean, optional, default=False
            Only has an effect when `self.val_df` is a list of DataFrame. If True, use the 
            method `Dataset.concatenate` to combine the data. Useful when plotting the 
            predictions. If False, use `Dataset.sample_from_datasets` to interleave the data.
        """
        if shuffle is None:
            shuffle = self.shuffle
        return self.make_dataset(self.val_df, self.batch_size * 2, shuffle, concat)

    def test_set(self, shuffle=None, concat=False):
        """
        Create windowed test set containing batches of sequences.

        Parameters
        ----------
        shuffle: boolean, optional, default=None
            Whether to shuffle output samples, or instead draw them in chronological order.
            If None, use `self.shuffle` defined during initialization.

        concat: boolean, optional, default=False
            Only has an effect when `self.test_df` is a list of DataFrame. If True, use the 
            method `Dataset.concatenate` to combine the data. Useful when plotting the 
            predictions. If False, use `Dataset.sample_from_datasets` to interleave the data.
        """
        if shuffle is None:
            shuffle = self.shuffle
        return self.make_dataset(self.test_df, self.batch_size * 2, shuffle, concat)

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.x_indices}",
                f"Output indices: {self.y_indices}",
                f"Input column name(s): {self.x_columns}",
                f"Output column name(s): {self.y_columns}",
            ]
        )
