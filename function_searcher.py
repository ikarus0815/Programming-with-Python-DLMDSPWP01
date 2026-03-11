"""Logic for selecting best‑fit ideal functions based on training data.
"""


import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


class SelectionError(Exception):
    """Raised when an ideal function cannot be selected (e.g. mismatched x)."""


@dataclass
class SelectionResult:
    ideal_index: int
    sum_sq: float
    max_dev: float


class FunctionSearcher:
    """Finds the optimal ideal functions for each training series."""

    def __init__(self) -> None:
        # mapping from training column name -> SelectionResult
        self.results: dict[str, SelectionResult] = {}

    def select_ideal_functions(
        self, training_df: pd.DataFrame, ideal_df: pd.DataFrame
    ) -> dict[str, SelectionResult]:
        """Choose the best ideal column for each training column.

        Parameters
        ----------
        training_df : pd.DataFrame
            DataFrame with columns ['x','y1','y2','y3','y4']
        ideal_df : pd.DataFrame
            DataFrame with columns ['x','y1',..., 'y50']

        Returns
        -------
        dict[str, SelectionResult]
            Mapping from training column name to selection result.

        Raises
        ------
        SelectionError
            If the x-values do not match between dataframes.
        """
        # ensure x-values align exactly
        if not training_df['x'].equals(ideal_df['x']):
            raise SelectionError('x-values in training and ideal dataframes do not match')

        self.results.clear()

        ideal_columns = [c for c in ideal_df.columns if c != 'x']

        # OPTIMIZATION: Pre-compute ideal values to avoid redundant to_numpy() calls
        # This eliminates 200 unnecessary array allocations (4 training cols × 50 ideal functions)
        ideal_values = {col: ideal_df[col].to_numpy(dtype=float) for col in ideal_columns}

        for train_col in training_df.columns:
            if train_col == 'x':
                continue
            best: SelectionResult | None = None
            train_vals = training_df[train_col].to_numpy(dtype=float)
            for idx, ideal_col in enumerate(ideal_columns, start=1):
                # OPTIMIZATION: Use cached ideal values instead of repeated to_numpy()
                ideal_vals = ideal_values[ideal_col]
                residuals = train_vals - ideal_vals
                # OPTIMIZATION: Use dot product for better numerical stability and slight performance gain
                sum_sq = float(np.dot(residuals, residuals))
                max_dev = float(np.max(np.abs(residuals)))
                if best is None or sum_sq < best.sum_sq:
                    # make sure we record a plain Python int (not np.int64 or float)
                    best = SelectionResult(ideal_index=int(idx), sum_sq=sum_sq, max_dev=max_dev)
            assert best is not None
            self.results[train_col] = best
        return self.results

    def get_threshold(self, train_column: str) -> float | None:
        """Return the mapping threshold for a given training column.

        The threshold is the maximum deviation observed during training
        multiplied by sqrt(2), as per the specification.
        """
        res = self.results.get(train_column)
        if res is None:
            return None
        return res.max_dev * math.sqrt(2)
