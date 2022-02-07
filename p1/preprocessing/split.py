#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, split.py

This module provides functions to split data into K folds and split training-validation into separate training and
validation sets.

"""
# Third party libraries
import pandas as pd


def make_splits(data: pd.DataFrame, problem_class: str, label_col: str, k_folds: int,
                val_frac: float = None) -> pd.DataFrame:
    """
    Assign each observation to a fold or split a train-validation set into train & validation sets.
    :param data: Dataframe to split
    :param problem_class: 'classification' or 'regression'; if 'classification', data stratified by class weight
    :param label_col: Label column used for stratified splitting
    :param k_folds: Number of folds for k_folds splitting
    :param val_frac: Validation fraction for train-validation splitting
    :return: DataFrame of splits
    K folds and train-validation splitting are mutually exclusive operations; one or the other
    must be specified or else the function will raise a ValueError.
    """
    validate_split_inputs(problem_class, k_folds, val_frac)

    # If the problem class is classification, extract the labels into a list
    if problem_class == "classification":
        labels = data[label_col].unique().tolist()

    # Else if the problem class is regression, create a dummy label of length 1
    else:
        labels = ["foo"]

    # Iterate over each label and perform splits stratified by class weight
    folds_li = []
    for label in labels:
        frame = data.copy()

        # For classification problem, we want to split the data by class
        if problem_class == "classification":
            mask = frame[label_col] == label
            frame = frame[mask]

        # Create a temp_id for binning
        frame["temp_id"] = list(range(len(frame)))

        # If splitting into k folds, split the data into k equal sized parts
        if k_folds:
            fold_label_size = len(frame) // k_folds
            bins = [x * fold_label_size for x in range(k_folds + 1)]
            bins[-1] = frame["temp_id"].max()
            name = "fold"

        # If splitting train and validation, split data based on val_frac
        else:
            val_right_index = int(len(frame) * val_frac)
            bins = [frame["temp_id"].min(), val_right_index, frame["temp_id"].max()]
            name = "train"

        # Cut the dataframe into bins
        frame = pd.cut(frame["temp_id"], bins=bins, include_lowest=True).cat.codes.rename(name).to_frame()

        # Let folds be 1-indexed and validation / train be 0-indexed
        if not val_frac:
            frame = frame + 1

        # Append the dataframe to the list of folds
        folds_li.append(frame)

    # Concatenate the dataframes into one
    folds_data: pd.DataFrame = pd.concat(folds_li)

    return folds_data


def split_train_val(data: pd.DataFrame, problem_class: str, label_col: str, val_frac: float,
                    random_state: int) -> tuple:
    """
    Split the train-validation set into separate train and validation sets.
    :param data: Dataframe of features and labels
    :param problem_class: 'classification' or 'regression'
    :param label_col: Column whose values we want to estimate
    :param val_frac: Fraction of train-validation to split into validation set
    :param random_state: Random state used to shuffle data
    :return: train, validation tuple
    """
    splits = make_splits(data, problem_class, label_col, k_folds=None, val_frac=val_frac)
    splits = splits.sample(frac=1, random_state=random_state)
    train = splits[splits["train"] == 1]
    val = splits[splits["train"] == 0]
    return train.join(data), val.join(data)


def validate_split_inputs(problem_class: str, k_folds: int, val_frac: float):
    """
    Check if the split inputs are valid.
    :param problem_class: 'classification' or 'regression'
    :param k_folds: Number of K folds
    :param val_frac: Validation fraction of training-validation set
    Either k_folds or val_frac must be None, and both cannot be None
    """
    if k_folds is not None and val_frac is not None:
        raise ValueError("Both k_folds and val_frac cannot be set; one must be None.")
    if k_folds is None and val_frac is None:
        raise ValueError("Both k_folds and val_frac cannot be None.")
    if not isinstance(k_folds, int) and val_frac is None:
        raise TypeError("k_folds argument must be an integer.")
    if val_frac is None and k_folds < 2:
        raise ValueError("k_folds must be greater than 1.")
    if problem_class not in ["classification", "regression"]:
        msg = "problem_class is {problem_class} but must be either 'classification' or 'regression'."
        raise ValueError(msg)
