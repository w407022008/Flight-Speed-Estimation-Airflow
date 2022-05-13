import math
import pandas as pd
from itertools import groupby

import tensorflow as tf

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt


def plot_all(
    df,
    cols=4,
    rows=None,
    title=None,
    height=None,
    width=None,
    properties=None,
    horizontal_spacing=0.2,
    vertical_spacing=0.15,
    skip_begin=0.0,
    skip_end=1.0,
    engine="matplotlib",
):
    """Visualize the data. Note that the properties with the same predix will be plotted
        in the same figure.

        Parameters
        ----------
        df: pandas.DataFrame
            The data to be visualized.

        cols: int, optional, default=4
            The number of columns.

        rows: int, optional, default=None
            The number of rows. If None, inferred as ceil(len(properties) / cols)
        
        title: str, optional, default=None
            The main title.

        height: int, optional, default=None
            The height of each subfigure. If `engine` is plotly, the default value is 400.
            If `engine` is matplotlib, the default value is 4. 

        width: int, optional, default=None
            The width of each subfigure. If `engine` is plotly, the default value is 400.
            If `engine` is matplotlib, the default value is 4.

        properties: list of str
            The properties to be plotted. If None, plot all properties.

        horizontal_spacing: float, optional, default=0.2
            The horizontal space between subfigures. Only takes effect when `engine` is plotly.

        vertical_spacing: float, optional, default=0.15
            The vertical space between subfigures. Only takes effect when `engine` is plotly.

        skip_begin: float, optional, default=0
            How much to skip from the beginning.

        skip_end: float, optional, default=1
            How much to skip from the end.
        
        engine: str, default="matplotlib"
            The backend engine used to plot figures, "plotly" or "matplotlib".
        """
    if isinstance(skip_begin, float):
        assert (
            0 <= skip_begin <= 1
        ), "If skip_begin is a float, it must be between 0 and 1."
        skip_begin = int(df.shape[0] * skip_begin)

    if isinstance(skip_end, float):
        assert 0 <= skip_end <= 1, "If skip_end is a float, it must be between 0 and 1."
        skip_end = int(df.shape[0] * skip_end)

    if height is None:
        height = 400 if engine == "plotly" else 4
    if width is None:
        width = 400 if engine == "width" else 4

    groups = {
        key: list(columns)
        for key, columns in groupby(df.columns, lambda column: column.partition("[")[0])
    }

    if properties is None:
        properties = list(groups)

    if cols is not None:
        rows = math.ceil(len(properties) / cols)
    elif rows is not None:
        cols = math.ceil(len(properties) / rows)
    else:
        raise ValueError("Must specify one of cols and rows.")

    if engine == "plotly":
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes="all",
            horizontal_spacing=horizontal_spacing / cols,
            vertical_spacing=vertical_spacing / rows,
            subplot_titles=properties,
            specs=[[{"type": "xy"} for c in range(cols)] for r in range(rows)],
        )

        for i, property in enumerate(properties):
            r = int(i / cols)
            c = i - r * cols
            subfigs = px.line(df[groups[property]])
            for subfig in subfigs.data:
                fig.add_trace(subfig, row=r + 1, col=c + 1)
            fig.update_xaxes(title_text="Time series", row=r + 1, col=c + 1)
        # Update font, height and width
        fig.update_layout(
            dict(
                title_text=title,
                font=dict(family="Arial", size=14, color="black"),
                showlegend=False,
                height=height * rows,
                width=width * cols,
            )
        )
    elif engine == "matplotlib":
        fig, axs = plt.subplots(
            nrows=rows,
            ncols=cols,
            sharex=True,
            squeeze=False,
            figsize=(width * cols, height * rows),
            tight_layout=True,
        )

        for i, property in enumerate(properties):
            r = int(i / cols)
            c = i - r * cols
            axs[r, c].plot(df[groups[property]])
            axs[r, c].set_title(property)
            axs[r, c].set_xlabel("Time series")
            axs[r, c].legend([0, 1, 2][: len(groups[property])])
        fig.suptitle(title)
    else:
        raise ValueError(f"Gotten unsupported engine {engine}.")
    return fig


def concat_df(df, length):
    """Concatenate a list of DataFrame and disgard the first `length` rows.
        """
    if isinstance(df, (tuple, list)):
        return pd.concat(
            [data.iloc[length - 1 :, :] for data in df], axis=0, ignore_index=True
        )
    else:
        return df.iloc[length - 1 :, :]


def visualize_predictions(
    window,
    plot_col,
    model=None,
    title=None,
    height=None,
    width=None,
    horizontal_spacing=0.2,
    vertical_spacing=0.15,
    residual=False,
    engine="matplotlib",
):
    """Visualize the predictions of neural networks.

        Parameters
        ----------
        window: WindowGenerator

        plot_col: str
            The column to be plotted.

        model: keras.Model, optional, default=None
            Neural networks.
        
        title: str, optional, default=None
            The main title.

        height: int, optional, default=None
            The height of each subfigure. If `engine` is plotly, the default value is 400.
            If `engine` is matplotlib, the default value is 4. 

        width: int, optional, default=None
            The width of each subfigure. If `engine` is plotly, the default value is 400.
            If `engine` is matplotlib, the default value is 4.

        horizontal_spacing: float, optional, default=0.2
            The horizontal space between subfigures. Only takes effect when `engine` is plotly.

        vertical_spacing: float, optional, default=0.15
            The vertical space between subfigures. Only takes effect when `engine` is plotly.
        
        residual: bool, default=False
            If True, the difference between the ground truth and the prediction will be plotted.
        
        engine: str, default="matplotlib"
            The backend engine used to plot figures, "plotly" or "matplotlib".
        """

    if height is None:
        height = 400 if engine == "plotly" else 4
    if width is None:
        width = 400 if engine == "width" else 4

    df = [
        concat_df(window.train_df, window.total_window_size),
        concat_df(window.val_df, window.total_window_size),
        concat_df(window.test_df, window.total_window_size),
    ]
    # set `shuffle` to False and `concat` to True to draw samples in chronological order
    inputs = [
        window.train_set(shuffle=False, concat=True),
        window.val_set(shuffle=False, concat=True),
        window.test_set(shuffle=False, concat=True),
    ]
    y_index = window.y_columns_indices.get(plot_col, None)

    subtitles = ["Training set", "Validation set", "Test set"]
    rows = 3 if residual else 1
    cols = 3

    if engine == "plotly":
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            horizontal_spacing=horizontal_spacing / 3,
            vertical_spacing=vertical_spacing,
            subplot_titles=subtitles,
            specs=[[{"type": "xy"} for c in range(cols)] for r in range(rows)],
        )
        for i in range(cols):
            true = df[i][plot_col]
            fig.add_trace(
                go.Scatter(y=true, mode="lines", name="Ground truth"), row=1, col=i + 1
            )
            if model is not None:
                pred = tf.squeeze(model.predict(inputs[i])[:, -1, y_index])
                fig.add_trace(
                    go.Scatter(y=pred, mode="lines", name="Prediction"),
                    row=1,
                    col=i + 1,
                )
                if residual:
                    diff = pred - true
                    fig.add_trace(
                        go.Scatter(y=diff, mode="lines", name="Error"), row=2, col=i + 1
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=tf.cumsum(diff * 0.005),
                            mode="lines",
                            name="Cumulative error",
                        ),
                        row=3,
                        col=i + 1,
                    )
            # update scenes
            fig.update_xaxes(title_text="Time series", row=rows, col=i + 1)
        # update font, height and width
        fig.update_layout(
            dict(
                title_text=title,
                font=dict(family="Arial", size=12, color="black"),
                showlegend=True,
                height=height * rows,
                width=width * 3,
            )
        )
    elif engine == "matplotlib":
        fig, axs = plt.subplots(
            nrows=rows,
            ncols=cols,
            sharex="col",
            squeeze=False,
            figsize=(width * 3, height * rows),
            tight_layout=True,
        )
        for i in range(cols):
            true = df[i][plot_col]
            axs[0, i].plot(true)
            axs[0, i].set_title(subtitles[i])
            if model is not None:
                pred = tf.squeeze(model.predict(inputs[i])[:, -1, y_index])
                axs[0, i].plot(pred)
                if residual:
                    diff = pred - true
                    axs[1, i].plot(diff)
                    axs[2, i].plot(tf.cumsum(diff * 0.005))
            axs[0, i].legend(["Ground truth", "Prediction"])
            axs[1, i].legend(["Error"])
            axs[2, i].legend(["Cumulative error"])
            axs[rows - 1, i].set_xlabel("Time series")
        fig.suptitle(title)
    else:
        raise ValueError(f"Gotten unsupported engine {engine}.")
    return fig


def get_predictions(window, model):
    """Get the predictions of neural networks for the training, validation and test sets.

        Parameters
        ----------
        window: WindowGenerator

        model: keras.Model
            Neural networks.
        """

    true = [
        concat_df(window.train_df, window.total_window_size),
        concat_df(window.val_df, window.total_window_size),
        concat_df(window.test_df, window.total_window_size),
    ]
    true = [df[window.y_columns] for df in true]
    # set `shuffle` to False and `concat` to True to draw samples in chronological order
    inputs = [
        window.train_set(shuffle=False, concat=True),
        window.val_set(shuffle=False, concat=True),
        window.test_set(shuffle=False, concat=True),
    ]
    pred = [model.predict(i, batch_size=window.batch_size)[:, -1, :] for i in inputs]
    pred = [pd.DataFrame(p, columns=window.y_columns) for p in pred]
    return true, pred
