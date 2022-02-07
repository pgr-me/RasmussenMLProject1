# Peter Rasmussen, Programming Assignment 1

This Python 3 program trains simple majority predictors across six datasets, using the mode as the estimator for classification datasets and the mean for regression datasets.

## Getting Started

The package is designed to be executed as a module from the command line. The user must specify the
 output directory as illustrated below. The PRasmussenAlgospa2/resources
directory provides example output files - which echo the dynamically-generated input - for the user.

```shell
python -m path/to/p1  -i path/to/in_dir -o path/to/out_dir/ -k <folds> -v <val frac> -r <random state>
```

As an example:
```shell
python -m path/to/p1  -i path/to/in_dir -o path/to/out_dir/ -k 5 -v 0.1 -r 777
```

A summary of the command line arguments is below.

Positional arguments:

    -i, --src_dir               Input directory
    -o, --dst_dir               Output directory

Optional arguments:    

    -h, --help                 Show this help message and exit
    -k, --k_folds              Number of folds
    -v, --val_frac             Fraction of validation observations
    -r, --random_state         Provide pseudo-random seed

## Key parts of program
* run.py: Executes data loading, preprocessing, training, socring, and output creation.
* preprocessor.py: Preprocesses data: loading, imputation, discretization, and fold assignment. 
* majority_predictor.py: Trains and scores
  * Classification datasets are scored on the basis of accuracy
  * Regression datasets are scored on the basis of mean squared error

## Features

* Performance metrics for each run for each dataset.
* Tested on all six datasets.
* Outputs provided as two files: 1) CSV of performance metrics by fold and 2) CSV of performance metrics by dataset.
* Control over number of folds, validation fraction, and randomization.

## Output Files

See the ```output.csv``` and ```summary.csv``` files in the ```data/``` directory.

## Licensing

This project is licensed under the CC0 1.0 Universal license.
