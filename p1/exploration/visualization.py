#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, visualization.py

This module provides various visualiation functions.

"""
# Third party libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from p1.preprocessing import Preprocessor


def plot_histograms(preprocessor: Preprocessor, numeric_cols: list):
    """
    Plot histograms of numeric columns.
    :param preprocessor: Preprocessor object
    :param numeric_cols: List of numeric columns to plot
    """
    if not numeric_cols:
        print("No numeric columns to plot.")
        return

    # Instantiate figure object
    fig = make_subplots(rows=len(numeric_cols), cols=1)

    # Loop over each numeric column and add its trace to the figure
    for index, numeric_col in enumerate(numeric_cols, start=1):
        values = preprocessor.data[numeric_col].tolist()
        fig.append_trace(go.Histogram(x=values, name=numeric_col), row=index, col=1)

    # Update the layout and display the plot
    fig.update_layout(height=200 * len(numeric_cols), width=1000, title_text=preprocessor.dataset_name)
    fig.show()
