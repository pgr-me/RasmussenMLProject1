#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 1, preprocessing.py

This module provides the Preprocessor class.

"""
# Standard library imports
from pathlib import Path
from collections import defaultdict
import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p1.preprocessing.jenks import compute_two_break_jenks
from p1.preprocessing.split import make_splits


class Preprocessor:
    def __init__(self, dataset_name: str, dataset_meta: dict, data_dir: Path):
        self.dataset_name = dataset_name
        self.dataset_meta = dataset_meta
        self.data_dir = Path(data_dir)
        self.dataset_src: Path = self.data_dir / dataset_meta["data_filename"]
        self.names_meta = pd.DataFrame(self.dataset_meta["names_meta"]).set_index("name")
        self.names = list(self.names_meta.index.values)
        self.imputed_data: t.Union[pd.DataFrame, None] = None
        self.numeric_columns: list = None
        self.jenks_breaks: dict = {}
        self.discretize_dict: defaultdict(lambda: {})

    def __repr__(self):
        return f"{self.dataset_name} Loader"

    def compute_natural_breaks(self, numeric_cols: list = None, n_breaks=2, exclude_ordinal=True) -> pd.DataFrame:
        """
        Compute two-class natural Jenks breaks for each numeric column.
        :param numeric_cols: List of numeric columns to compute breaks for
        :param n_breaks: Number of breaks to split list into
        :param exclude_ordinal: True to exclude ordinal columns
        :return: Dataframe of indexed break assignments
        """
        if n_breaks != 2:
            msg = "Jenks breaks are only available for two classes / breaks."
            raise NotImplementedError(msg)

        # Select all numeric columns if none are provided
        numeric_cols = self.get_numeric_columns() if numeric_cols is None else numeric_cols

        # If indicated, remove ordinal columns
        if exclude_ordinal:
            ordinal_cols = self.names_meta[self.names_meta.data_class == "ordinal"].index
            numeric_cols = [x for x in numeric_cols if x not in ordinal_cols]

        for numeric_col in numeric_cols:
            values = self.data[numeric_col].tolist()
            self.jenks_breaks[numeric_col] = compute_two_break_jenks(values)

        self.jenks_breaks = pd.DataFrame.from_dict(self.jenks_breaks).transpose()
        self.jenks_breaks.sort_values(by="gcvf", ascending=False, inplace=True)
        return self.jenks_breaks

    def discretize(self, discretize_dict: dict) -> pd.DataFrame:
        """
        Discretize indicated columns using provided discretize_dict.
        :param discretize_dict: Dictionary keyed by column
        :return: Discretized columns
        Example discretize_dict structure:
            {"bare_nuclei": {"n_bins": 2, "binning": "equal_width"},
             "normal_nucleoli": {"n_bins": 2, "binning": "equal_width"}}
        """
        self.discretize_dict = defaultdict(lambda: {}, discretize_dict)
        for col, bin_dict in self.discretize_dict.items():
            frame, retbins = self._discretize(self.data[col], bin_dict["n_bins"], bin_dict["binning"])
            self.data.drop(axis=1, labels=col, inplace=True)
            self.data = self.data.join(frame)
            self.discretize_dict[col]["retbins"] = retbins
        return self.data[list(discretize_dict.keys())]

    def dummy(self, columns: t.Union[list[str], str, None] = "default") -> pd.DataFrame:
        """
        Dummy categorical columns.
        :param columns: 'default' for defaults, list to specify them, False / None to do nothing
        :return: Data
        """
        if columns == "default":
            mask = self.names_meta["data_class"] == "categorical"
            columns = self.names_meta[mask].index.values.tolist()
        if columns:
            self.data = pd.get_dummies(self.data, columns=columns)

        # Update features list
        self.features = [x for x in self.data if (x not in self.label) and (x not in self.index)]
        index_cols = self.names_meta[self.names_meta["id"]].index.values
        self.features = [x for x in self.data if x not in index_cols]
        return self.data

    def get_numeric_columns(self, exclude_index=True):
        """
        Retrieve numeric columns using names metadata.
        :return: List of numeric columns
        """
        mask = self.names_meta["data_type"].isin(["int", "float"])
        if exclude_index:
            mask = mask & (self.names_meta["id"] == False)
        self.numeric_columns = self.names_meta[mask].index.tolist()
        return self.numeric_columns

    def identify_features_label_id(self) -> pd.DataFrame:
        """
        Parse features, label, and ID columns from metadata.
        :return: Modified dataframe
        """
        # Identify features, label, and id columns
        mask = self.names_meta["feature"]
        self.features: list = self.names_meta[mask].index.values.tolist()
        mask = self.names_meta["label"]
        self.label: str = self.names_meta[mask].index.values[0]

        self.data["index"] = list(range(len(self.data)))
        self.index = "index"

        # Set the index
        self.data[self.index] = self.data[self.index].astype(int)
        self.data.set_index(self.index, inplace=True)

        return self.data

    def impute(self, numeric_cols: t.Union[list[str], str] = "default", strategy: str = "mean"):
        """
        Impute missing values of numeric columns.
        :param strategy: Currently only mean is implemented
        """
        if strategy != "mean":
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        data = self.data.copy()
        if numeric_cols == "default":
            numeric_cols = list(data.select_dtypes(np.number))
        for col in numeric_cols:
            self.data[col] = self._impute(data[col], strategy=strategy)
        return self.data

    def load(self) -> pd.DataFrame:
        """
        Load CSV of dataset for Projects 1 to 4 into a dataframe.
        :return: Loaded dataset
        """
        # Set column names, replace missing values with NaNs, and set data types
        na_values = self.dataset_meta["missing"]
        dtypes = self.make_dtypes(self.names_meta.data_type.to_dict())
        kwargs = {"names": self.names,
                  "na_values": na_values,
                  "dtype": dtypes}
        if self.dataset_meta["header"]:
            kwargs.update({"header": 0})
        self.data = pd.read_csv(self.dataset_src, **kwargs)
        return self.data

    def log_transform(self, log_transforms: t.Union[list[str], str, bool] = "default") -> pd.DataFrame:
        """
        Log transform indicated columns.
        :param log_transforms: 'default' for defaults, list to specify them, False to do nothing
        :return: Data
        """
        # If "default", use default log transformations from data catalog
        if log_transforms == "default":
            mask = self.names_meta["log_transform"]
            log_transforms = self.names_meta["log_transform"][mask].index.tolist()

        # Perform log transformations
        if log_transforms:
            for col in log_transforms:
                self.data[col] = np.log(self.data[col])
        return self.data

    def make_folds(self, k_folds: int):
        """
        Make folds and add them to dataset.
        :param k_folds: Number of folds to create
        :return: Folds dataframe
        """
        problem_class = self.dataset_meta["problem_class"]
        folds: pd.DataFrame = make_splits(self.data, problem_class, self.label, k_folds)
        self.data = folds.join(self.data)
        return folds

    def replace(self, replace_di: t.Union[dict, str, None] = "default") -> pd.DataFrame:
        """
        Replace dataframe values for indicated columns.
        :param replace_di: 'default' for defaults, dict for custom, None to do nothing
        :return: Transformed (or not) dataframe
        This function doubles as the one that converts ordinal string data to integers.
        """
        # If "default", use default replacements from data catalog
        if replace_di == "default":
            replace_di = self.names_meta["replace"].dropna().to_dict()

        # If True or dict, replace values for indicated columns
        if replace_di:
            for col, di in replace_di.items():
                self.data[col] = self.data[col].replace(di)
        return self.data

    def shuffle(self, random_state: int = 777) -> pd.DataFrame:
        """
        Shuffle the data by random seed.
        """
        self.data = self.data.sample(frac=1, random_state=random_state)
        return self.data

    @staticmethod
    def _discretize(series: pd.Series, n_bins: int, binning: str = "equal_frequency") -> tuple:
        """
        Discretize a numeric series into categorical or ordinal bins.
        :param series: Numeric series to discretize
        :param n_bins: Number of bins resulting from discretization
        :param binning: Binning strategy used for discretization
        :param retbins: True to return bin definitions
        :return: Tuple of two elements: Discretized dataframe and bin definitions
        For the 'equal_frequency' binning strategy, retbins is None
        """
        # Sort by values - necessary for equal_frequency method
        series = series.copy().sort_values()
        name = series.name

        # Make bins based on binning strategy and cut the data
        if binning == "equal_width":
            min_, max_ = series.min(), series.max()
            range_ = max_ - min_
            increment = range_ / n_bins
            bins = [min_ + x * increment for x in range(n_bins + 1)]
            cuts, retbins = pd.cut(series, bins=bins, include_lowest=True, retbins=True)
            cuts = cuts.cat.codes
        elif binning == "equal_frequency":
            bin_size = len(series) / n_bins
            cuts = pd.Series([int(x // bin_size) for x in range(len(series))], index=series.index)
            retbins = None
        else:
            raise ValueError(f"{binning} binning is not supported / unknown to this implementation.")

        # Reset series name
        cuts.name = name

        return cuts.to_frame(), retbins

    @staticmethod
    def _impute(feature: pd.Series, strategy: str) -> pd.Series:
        """
        Impute missing feature values using the selected strategy.
        :param feature: Feature values
        :param strategy: Imputation strategy
        :return: Imputed feature series
        The only strategy currently implemented is "mean".
        """
        if not isinstance(feature, pd.Series):
            raise TypeError("Feature must be a Pandas series.")

        # Convert infinitely large or small values to NaNs
        mask = feature.copy().isin([float("inf"), -float("inf"), np.inf, -np.inf])
        feature.loc[mask] = np.nan

        # Impute missing values with the feature's mean
        if strategy == "mean":
            feature = pd.to_numeric(feature, errors="coerce", downcast="float")
            feature = feature.fillna(feature.mean())
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")

        return feature

    @staticmethod
    def make_dtypes(dtypes_di: dict[str]) -> dict[str]:
        """
        Prepare input dtype mapping for load_data method.
        :return: Prepared dtype mapping
        """
        return {k: v.replace("int", "float") for k, v in dtypes_di.items()}
