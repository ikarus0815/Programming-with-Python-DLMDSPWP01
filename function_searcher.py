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
        # Extract ALL ideal column data at once (200 calls reduced to 50)
        # This eliminates 200 unnecessary array allocations (4 training cols × 50 ideal functions)
        ideal_values = {}
        for ideal_col in ideal_columns:
            # Use .values for faster extraction than .to_numpy()
            ideal_values[ideal_col] = ideal_df[ideal_col].values.astype(float, copy=False)

        # Pre-compute all training values as well
        training_columns = [c for c in training_df.columns if c != 'x']
        training_values = {col: training_df[col].values.astype(float, copy=False) 
                          for col in training_columns}

        for train_col in training_columns:
            best: SelectionResult | None = None
            # Use pre-computed training values
            train_vals = training_values[train_col]
            
            for idx, ideal_col in enumerate(ideal_columns, start=1):
                # OPTIMIZATION: Use cached ideal values instead of repeated to_numpy()
                ideal_vals = ideal_values[ideal_col]
                
                # OPTIMIZATION: Use dot product for absolute maximum performance
                # Equivalent to sum((a-b)^2) but optimized for BLAS operations
                residuals = train_vals - ideal_vals
                sum_sq = float(np.dot(residuals, residuals))
                
                # Alternative: compute max deviation while we have residuals
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
