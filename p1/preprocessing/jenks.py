#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, jenks.py

This module provides functions to split data using Jenks natural breaks method.

"""
# Standard library imports
import typing as t

# Third party libraries
import pandas as pd


def compute_gcvf(sdam: float, scdm: float) -> float:
    """
    Compute the goodness of class variance fit (GCVF).
    :param sdam: Sum of squared deviations for array mean (SDAM)
    :param scdm: Sum of squared class devations from mean (SCDM)
    :return: GCVF
    Sources:
        https://arxiv.org/abs/2005.01653
        https://medium.com/analytics-vidhya/jenks-natural-breaks-best-range-finder-algorithm-8d1907192051
    """
    return (sdam - scdm) / sdam


def compute_sdam(values: list, mean: t.Union[int, float] = None):
    """
    Compute the sum of squared deviations for array mean.
    :param values: List of values
    :param mean: Mean of values
    Sources:
        https://arxiv.org/abs/2005.01653
        https://medium.com/analytics-vidhya/jenks-natural-breaks-best-range-finder-algorithm-8d1907192051
    """
    # Compute mean if not provided
    if mean is None:
        mean = sum(values) / len(values)
    return sum([(x - mean) ** 2 for x in values])


def compute_two_break_jenks(values: list[int, float]) -> dict:
    """
    Compute two-class Jenks break for a vector of numeric values.
    :param values: List of numeric values
    :return: Dictionary keyed by break value and goodness of variance fit (GCVF)
    Sources:
        https://arxiv.org/abs/2005.01653
        https://medium.com/analytics-vidhya/jenks-natural-breaks-best-range-finder-algorithm-8d1907192051
    """
    # Make sure values are sorted
    values = sorted(values)

    # Compute sum of squared deviations for dataset mean (SDAM)
    sdam = compute_sdam(values)
    scdm_li = []
    for index in range(1, len(values) - 1):
        # Compute sum of squared deviations for class means (SCDM)
        left_cut = values[:index]
        right_cut = values[index:]
        scdm = compute_sdam(left_cut) + compute_sdam(right_cut)
        scdm_li.append([index, scdm])

    # Select index and scdm corresponding to minimum scdm
    cols = ["index", "scdm"]
    selected = pd.DataFrame(scdm_li, columns=cols).set_index("index").sort_values(by="scdm").iloc[0, :]
    selected_index, selected_scdm = selected.name, selected["scdm"]
    break_value = values[selected_index]

    # Compute goodness of class variance fit (GCVF)
    gcvf = compute_gcvf(sdam, selected_scdm)

    return {"break_value": break_value, "gcvf": gcvf}