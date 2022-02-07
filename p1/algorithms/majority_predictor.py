#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, preprocessing.py

This module provides the Preprocessor class.

"""
# Third party libraries
import numpy as np
import pandas as pd


class MajorityPredictor:
    """
    This class predicts the mode and mean for classification and regression problems, respectively.
    """

    def __init__(self, problem_class: str, label_col: str, feature_cols: list[str]):
        """
        Instantiate the MajorityPredictor object.
        """
        self.problem_class = problem_class
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.beta: float = None
        self.pred: pd.Series = None
        self.score_: float = None
        self.X_tr: pd.DataFrame = None

        if self.problem_class not in ["classification", "regression"]:
            raise ValueError("Problem class must be either 'classification' or 'regression'.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Train the predictor on the data: take mode if classification else take mean.
        :param X_train: Training feature values
        :param y_train: Training label values
        """
        self.X_train = X_train
        self.y_train = y_train
        if self.problem_class == "classification":
            self.beta = self.y_train.mode().loc[0]
        else:
            try:
                self.beta = self.y_train.mean().loc[0]
            except AttributeError:
                self.beta = self.y_train.mean()
        return self.beta

    def tune(self, X_train, X_tune, y_train, y_tune):
        """
        Use the tuning data to improve the predictive power of the model.
        :param X_train: Training feature values
        :param X_tune: Tuning feature values
        :param y_train: Training label values
        :param y_tune: Tuning label values
        """
        X = pd.concat([X_train, X_tune])
        y = pd.concat([y_train, y_tune])
        self.train(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict X using beta.
        :param X: Dataframe of feature values
        :return Predicted label values
        """
        return pd.Series([self.beta for x in range(len(X))], index=X.index, name="pred")

    def score(self, y_pred: pd.Series, y_truth: pd.Series) -> float:
        """
        Score outputs using sum of squared error.
        :param y_pred: Predicted y
        :param y_truth: True y
        :return: Score
        If regression compute MSE, otherwise estimate accuracy.
        """

        if self.problem_class == "regression":
            self.score_ = np.sum(y_pred - y_truth) ** 2 / len(y_truth)
        else:
            self.score_ = (y_truth == y_pred).sum() / len(y_truth)
        return self.score_

