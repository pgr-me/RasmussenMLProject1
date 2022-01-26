"""Peter Rasmussen, Programming Assignment 2, __main__.py

This program selects finds the longest interweaving of signals of x and y, if it exists, in a
candidate interweaving. Characters are limited to binary integers and an error is raised if any
other values are encountered.

Inputs: Inputs are created dynamically using a random seed that the user provides and an upper limit
of candidate interweaving string length.

Outputs: Two outputs are generated and saved to a user-specified directory. The first is
io_data.csv. An example is saved in the resources directory for the default seed of 777. The second
output is is summary_statistics.csv, an example of which is also saved in the resources directory
using the same seed. *Please note the input data that is dynamically generated IS saved in the
io_data.csv*.

The structure of this package is based on the Python lab0 package that Scott Almes developed for
Data Structures 605.202. Per Scott, this module "is the entry point into this program when the
module is executed as a standalone program."

Please note the default Python recursion depth needed to be modified (see line 33 below) so that the
maximum recursion depth would not be exceeded.

"""

# standard library imports
import argparse
from pathlib import Path
import sys

# local imports
from pa2.run import run


sys.setrecursionlimit(16000)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--number", "-n", default=2**10, type=Path, help="Max length of s"
)
parser.add_argument(
    "--dst_dir", "-o", type=Path, help="Output directory"
)
parser.add_argument(
    "--seed", "-s", default=777, type=int, help="Pseudo-random seed"
)
args = parser.parse_args()

run(
    args.number,
    args.dst_dir,
    args.seed
)
