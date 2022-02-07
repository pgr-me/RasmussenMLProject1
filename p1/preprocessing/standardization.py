#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, standardization.py

This module provides data standardization functions.

"""
# Standard library imports
from pathlib import Path
from collections import defaultdict
import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p1.preprocessing.jenks import compute_two_break_jenks
from p1.preprocessing.split import make_splits


def get_standardization_cols(data: pd.DataFrame, feature_cols: list)->list:
    """
    Retrieve which columns are to be standardized.
    :param data: Dataframe to get standardization columns
    :param feature_cols: List of feature columns
    :return: List of columns to be standardized
    """
    # Exclude Boolean columns from standardization
    mask = (preprocessor.data.min() == 0) & (preprocessor.data.max() == 1)
    standardization_cols = [x for x in mask[~mask].index if x not in ["fold", "index", "train"] and x in feature_cols]
    return data[standardization_cols].columns.tolist()

def get_standardization_params(data: pd.DataFrame)->tuple:
    """
    Get the means and standard deviations of each specified column in a dataset.
    :param data: Dataframe to obtain standardization parameters from
    :return: Tuple of a means series and a standard deviation series
    """
    return data.mean(), data.std()

def standardize(data: pd.DataFrame, means: pd.Series, standard_deviations: pd.Series):
    """
    Standardize dataframe by subtracting mean and dividing by standard deviation.
    :param data: Dataframe to standardzie
    :param means: Series of means to standardize columns
    :param standard_deviations: Series of standard deviations to standardize columns
    """
    return data.subtract(means, axis="columns").div(standard_deviations, axis="columns")