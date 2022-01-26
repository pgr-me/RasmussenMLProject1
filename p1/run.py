"""Peter Rasmussen, Programming Assignment 2, run.py

The run function ingests user inputs of random seed and output directory and generates a set of
candidate interweavings (s), x signals, and y signals. Each s, x, and y tuple is organized into a
run. Each run is processed to find the longest interweaving of signals x and y, if they exist, in s.

Outputs are saved to the user-specified directory.

"""

# standard library imports
import csv
import logging
import math
import os
from pathlib import Path
import typing as t

# local imports
from pa2.datamaker import DataMaker
from pa2.graph import Graph
from pa2.utils import summarize, validate_candidate_interweaving


def run(
        n: int,
        dst_dir: Path,
        seed: int,

):
    """
    Symbolically combine polynomials and then evaluate for various evaluation sets.
    :param n: Max size of s
    :param dst_dir: Output directory
    :param seed: Random number seed
    """
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    log_path = dir_path / "pa2.log"
    io_data_dst = dst_dir / "io_data.csv"
    summary_stats_dst = dst_dir / "summary_stats.csv"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)

    logging.debug(f"Begin: n={n}, dst_dir={dst_dir.name}, seed={seed}")

    logging.debug("Generate data")
    n_li = []
    for n_ in range(int(math.log(n, 2))):
        n_li.append(2 ** n_)

    x_len_li = [2, 4, 8, 16, 32]
    y_len_li = [2, 4, 8, 16, 32]
    run_number = 0
    io_data = []
    summary_stats = []
    io_fields = ["s", "x", "y", "interweaving"]

    for s_len in n_li:
        s: t.List[int] = DataMaker(s_len, seed=s_len).make_data()
        logging.debug(f"Validate candidate interweaving for run={run_number} and s_len={s_len}")
        validate_candidate_interweaving(s)

        # For each combination of s, x, and y signal lengths, make data, process it, & write outputs
        for x_len in x_len_li:
            x: t.List[int] = DataMaker(x_len, seed=x_len * 2 + 2).make_data()
            for y_len in y_len_li:
                y: t.List[int] = DataMaker(y_len, seed=x_len * 4 + 2).make_data()
                if (x_len < s_len) and (y_len < s_len):
                    msg = f"Run {run_number} for s_len={s_len}, x_len={x_len}, y_len={y_len}"
                    logging.debug(msg)
                    G = Graph(s, x, y)
                    G.build_table()
                    G.build_graph()
                    G.find_max_rank_node()
                    G.find_longest_path()

                    summary = summarize(G, run_number)
                    io_data.append({k: v for k, v in summary.items()})
                    summary_stats.append({k: v for k, v in summary.items() if k not in io_fields})

                    run_number += 1

    logging.debug(f"Save IO data outputs to {io_data_dst.name}")
    with open(io_data_dst, 'w', newline='') as csv_file:
        fieldnames = list(io_data[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in io_data:
            writer.writerow(row)

    logging.debug(f"Save summary stats to {summary_stats_dst.name}")
    with open(summary_stats_dst, 'w', newline='') as csv_file:
        fieldnames = list(summary_stats[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_stats:
            writer.writerow(row)

    logging.debug("Finish.\n")
