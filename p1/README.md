# Peter Rasmussen, Programming Assignment 2

This Python 3 program finds the longest interweaving, if it exists, of strings x and y in a string s. 

## Getting Started

The package is designed to be executed as a module from the command line. The user must specify the
 output directory as illustrated below. The PRasmussenAlgospa2/resources
directory provides example output files - which echo the dynamically-generated input - for the user.

```shell
python -m path/to/pa2  -o path/to/out_dir/ 
```

Finally, the user may specify the random seed used to generate the randomly distributed set of
points, as the example below shows.

```shell
python -m path/to/pa2 -o path/to/out_dir/ -s 777
```

A summary of the command line arguments is below.

Positional arguments:

    -o, --dst_dir      Output directory

Optional arguments:    

    -h, --help         Show this help message and exit
    -n, --number       Max length of string s
    -s, --seed         Provide pseudo-random seed

## Key parts of program
* run module: Executes data generation, processing (i.e., finding interweaving), and post-processing.
* DataMaker: Generates a list of pseudo-random binary integers.
* Graph: Class that assimilates input data and finds the longest interweaving of x and y signals

## Features

* Performance metrics for each run for each algorithm: number of distance comparisons and total 
  number of operations organized by run into a CSV output.
* Tested on inputs of up to n=512 (recursion depth limits prevented larger inputs).
* Outputs provided as two files: 1) CSV of performance metrics and 2) data input / output (IO) that
  echoes inputs and provides full outputs, including interweaving of x and y.
* Control over randomization by selection of random seed.

## Output Files

See the two files in the ```resources/outputs``` directory.

## Licensing

This project is licensed under the CC0 1.0 Universal license.
