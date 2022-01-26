#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, utils.py

This module provides miscellaneous utility functions for this package.

"""
# Standard library imports
import typing as t

# Local imports
from pa2.graph import Graph
from pa2.signal_processing import count_signals


def summarize(G: Graph, run: int = 0) -> dict:
    """
    Summarize outputs of finding interweaving in G.
    :param G: Graph
    :param run: Run number
    :return: Inputs, outputs, and summary statistics
    """
    if len(G.longest_path) == 0:
        return {"run": run, "s": G.s, "x": G.x, "y": G.y, "s_len": len(G.s), "x_len": len(G.x),
                "y_len": len(G.y), "x_count": 0, "y_count": 0, "interweaving": [],
                "interweaving_len": 0, "n_ops": G.n_ops, "n_comps": G.n_comps}
    x_count = count_signals(G.longest_path, signal="x")
    y_count = count_signals(G.longest_path, signal="y")
    ranks = G.longest_path.keys()
    interweaving = [G.longest_path[rank]["s_char"] for rank in ranks]
    s = "".join([str(_) for _ in G.s])
    x = "".join([str(_) for _ in G.x])
    y = "".join([str(_) for _ in G.y])
    return {"run": run, "s": s, "x": x, "y": y, "s_len": len(G.s), "x_len": len(G.x),
            "y_len": len(G.y), "x_count": x_count, "y_count": y_count, "interweaving": interweaving,
            "interweaving_len": len(interweaving), "n_ops": G.n_ops, "n_comps": G.n_comps}


def validate_candidate_interweaving(s: t.List[str]):
    """
    Validate that candidate interweaving s only contains integers 0 and 1.
    :param s: Candidate interweaving
    """
    for ix, char in enumerate(s):
        if not isinstance(char, int):
            msg = f"Character {char} at index {ix} of candidate interweaving {s} is of type {type(char)} but should be int."
            raise TypeError(msg)
        if char not in [0, 1]:
            msg = f"Integer {char} at index {ix} of candidate interweaving {s} must be 0 or 1."
            raise ValueError(msg)
