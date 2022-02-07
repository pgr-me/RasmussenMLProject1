"""Peter Rasmussen, Programming Assignment 1, __main__.py

This program trains a simple majority predictor across six datasets to estimate the 1) class for
classification datasets or 2) mean for regression datasets. Classification datasets are scored on
accuracy and regression datasets are scored by mean squared error. Data is split into test, train,
and validation sets and the user specifies the number of folds to use.

Inputs: Six datasets obtained from the course website are used in this analysis. They are available in
the RasmussenMLProject1/data directory of this repo.

Outputs: Two outputs are generated and saved to a user-specified directory. The first is
output.csv. This provides more detailed, fold-level scoring and parameter (beta) outputs. The second,
summary.csv, provides dataset-level performance statistics.

The structure of this package is based on the Python lab0 package that Scott Almes developed for
Data Structures 605.202. Per Scott, this module "is the entry point into this program when the
module is executed as a standalone program."

"""

# standard library imports
import argparse
from pathlib import Path

# local imports
from p1.run import run


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--src_dir", "-i", type=Path, help="Input directory"
)
parser.add_argument(
    "--dst_dir", "-o", type=Path, help="Output directory"
)
parser.add_argument(
    "--k_folds", "-k", default=5, type=int, help="Number of folds to partition data"
)
parser.add_argument(
    "--val_frac", "-v", default=0.1, type=float, help="Fraction of validation samples"
)
parser.add_argument(
    "--random_state", "-r", default=777, type=int, help="Pseudo-random seed"
)
args = parser.parse_args()

run(
    args.src_dir,
    args.dst_dir,
    args.k_folds,
    args.val_frac,
    args.random_state
)
