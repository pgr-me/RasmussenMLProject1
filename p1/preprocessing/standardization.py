#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, standardization.py

This module provides data standardization functions.

"""
# Third party imports
import pandas as pd


def get_standardization_cols(data: pd.DataFrame, feature_cols: list) -> list:
    """
    Retrieve which columns are to be standardized.
    :param data: Dataframe to get standardization columns
    :param feature_cols: List of feature columns
    :return: List of columns to be standardized
    """
    # Exclude Boolean columns from standardization
    mask = (data.min() == 0) & (data.max() == 1)
    standardization_cols = [x for x in mask[~mask].index if x not in ["fold", "index", "train"] and x in feature_cols]
    return data[standardization_cols].columns.tolist()


def get_standardization_params(data: pd.DataFrame) -> tuple:
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
