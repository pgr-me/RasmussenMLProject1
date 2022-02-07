"""Peter Rasmussen, Programming Assignment 1, run.py

The run function ingests user inputs to train majority predictors on six different datasets.

Outputs are saved to the user-specified directory.

"""

# Standard library imports
from collections import defaultdict
import json
import logging
import os
from pathlib import Path

# Third party imports
import pandas as pd

# Local imports
from p1.preprocessing import Preprocessor, get_standardization_cols, get_standardization_params, standardize, split_train_val
from p1.algorithms import MajorityPredictor


def run(
        src_dir: Path,
        dst_dir: Path,
        k_folds: int,
        val_frac: float,
        random_state: int,
):
    """
    Train and score a majority predictor across six datasets.
    :param src_dir: Input directory that provides each dataset and params files
    :param dst_dir: Output directory
    :param k_folds: Number of folds to partition the data into
    :param val_frac: Validation fraction of train-validation set
    :param random_state: Random number seed

    """
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    log_path = dir_path / "p1.log"
    with open(src_dir / "discretize.json") as file:
        discretize_dicts = json.load(file)
    discretize_dicts = defaultdict(lambda: {}, discretize_dicts)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format=log_format)


    logging.debug(f"Begin: src_dir={src_dir.name}, dst_dir={dst_dir.name}, seed={random_state}.")

    with open(src_dir / "data_catalog.json", "r") as file:
        data_catalog = json.load(file)

    # Initialize the list to hold our outputs
    output = []

    # Loop over each dataset and its metadata using the data_catalog
    for dataset_name, dataset_meta in data_catalog.items():
        logging.debug(f"Load and process dataset {dataset_name}.")

        # Load data: Set column names, data types, and replace values
        preprocessor = Preprocessor(dataset_name, dataset_meta, src_dir)
        preprocessor.load()

        # Identify which columns are features, which is the label, and any ID columns
        preprocessor.identify_features_label_id()

        # Replace values: Ordinal strings (lower, higher) replace with numeric values
        preprocessor.replace()

        # Log transform indicated columns (default is to take selected columns from dataset_meta)
        preprocessor.log_transform()

        # Impute missing values
        preprocessor.impute()

        # Dummy categorical columns
        preprocessor.dummy()

        # Discretize indicated columns
        preprocessor.discretize(discretize_dicts[dataset_name])

        # Randomize the order of the data
        preprocessor.shuffle(random_state=random_state)

        # Make K folds and assign each observation to one
        preprocessor.make_folds(k_folds)

        # Extract dataframe from preprocessor object
        data = preprocessor.data.copy()

        # Define each column as a feature, label, or index
        feature_cols = preprocessor.features
        label_col = preprocessor.label
        problem_class = dataset_meta["problem_class"]  # regression or classification
        if problem_class == "classification":
            data[label_col] = data[label_col].astype(int)

        # Iterate over each fold
        for fold in range(1, k_folds + 1):
            # Split test and train-validation sets
            mask = data["fold"] == fold
            test = data.copy()[mask]
            train_val = data.copy()[~mask]

            # Get standardization parameters from training-validation set
            cols = get_standardization_cols(train_val, feature_cols)
            means, std_devs = get_standardization_params(data.copy()[cols])

            # Standardize data
            test = test.drop(axis=1, labels=cols).join(standardize(test[cols], means, std_devs))
            train_val = train_val.drop(axis=1, labels=cols).join(standardize(train_val[cols], means, std_devs))

            # Split train and validation sets
            train, val = split_train_val(train_val, problem_class, label_col, val_frac, random_state)

            # Instantiate the model object
            predictor = MajorityPredictor(problem_class, label_col, feature_cols)

            # Train the model
            predictor.train(train[feature_cols], train[label_col])

            # Predict
            predictor = MajorityPredictor(problem_class, label_col, feature_cols)
            predictor.train(train[feature_cols], train[label_col])
            predictor.tune(train[feature_cols], val[feature_cols], train[label_col], val[label_col])
            y_test_pred = predictor.predict(test)
            y_test_truth = test.copy()[label_col]
            test_score = predictor.score(y_test_pred, y_test_truth)
            output_li = [dataset_name, problem_class, fold, test_score, predictor.beta]
            output.append(output_li)
            logging.info(f"Dataset {dataset_name}: fold: {fold}, score: {test_score}.")

    logging.debug("Process outputs.")
    # Organize outputs
    output_df = pd.DataFrame(output, columns=["dataset_name", "problem_class", "fold", "test_score", "beta"])

    # Compute mean test score across folds for each dataset
    summary = output_df.groupby(["problem_class", "dataset_name"])["test_score"].mean().to_frame().round(2)

    # Save outputs
    logging.debug("Save outputs.")
    output_dst = dst_dir / "output.csv"
    summary_dst = dst_dir / "summary.csv"
    output_df.to_csv(output_dst)
    summary.to_csv(summary_dst)

    logging.debug("Finish.\n")
