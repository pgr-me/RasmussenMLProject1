#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, signal_processing.py

This module provides signal processing helper functions.

"""
from copy import deepcopy
import typing as t


def count_signals(path: dict, signal="x"):
    """
    Count number of x's or y's in a path.
    :param path: Dictionary of linked nodes
    :param signal: 'x' or 'y'
    :return: Number of x's or y's in path
    """
    if signal not in ["x", "y"]:
        raise ValueError(f"Signal {signal} must be either 'x' or 'y'.")
    return len([di["signal"] for rank, di in path.items() if di["signal"] == signal])


def get_s_ij(s: t.List[int], i: int, j: int) -> t.Union[int, None]:
    """
    Retrieve s char at i and j indices.
    :param s: Candidate interweaving
    :param i: Row index
    :param j: Column index
    :return: Character at i and j indices
    """
    if (i + j) < len(s):
        return s[(i + j) % len(s)]
    else:
        return None


def get_x_ij(x: t.List[int], s: t.List[int], i: int, j: int) -> t.Union[int, None]:
    """
    Get x char at index
    :param x: x signal
    :param s: Candidate interweaving
    :param i: Row index
    :param j: Column index
    :return: Char of x at row i and column j
    """
    if (i + j) < len(s):
        return x[j % len(x)]
    else:
        return None


def get_y_ij(y: t.List[int], s: t.List[int], i: int, j: int) -> t.Union[int, None]:
    """
    Get y char at index
    :param y: y signal
    :param s: Candidate interweaving
    :param i: Row index
    :param j: Column index
    :return: Char of y at row i and column j
    """
    if (i + j) < len(s):
        return y[i % len(y)]
    else:
        return None


def prune_path(path_: dict, x: t.List[int], y: t.List[int]) -> t.Tuple[dict, int]:
    """
    Prune path so that it only contains complete sequences of x and y.
    :param path_: Un-pruned path
    :param x: x signal
    :param y: y signal
    :return: Pruned path
    """
    path = deepcopy(path_)
    x_count = count_signals(path, signal="x")
    y_count = count_signals(path, signal="y")
    n_ops = 3  # Three for the above lines
    while (x_count % len(x) != 0) or (y_count % len(y) != 0):
        path.pop(max(path))
        x_count = count_signals(path, signal="x")
        y_count = count_signals(path, signal="y")
        n_ops += 5  # Five for the while loop lines and three lines below and the return statement
    return path, n_ops
