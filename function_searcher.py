"""Logic for selecting best‑fit ideal functions based on training data.
"""


import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba_kernels import find_best_ideal_by_column, NUMBA_AVAILABLE


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
        if not training_df['x'].equals(ideal_df['x']):
            raise SelectionError('x-values in training and ideal dataframes do not match')

        self.results.clear()

        ideal_columns = [c for c in ideal_df.columns if c != 'x']

        # OPTIMIZATION: Pre-compute ideal values to avoid redundant to_numpy() calls
        # Extract ALL ideal column data at once (200 calls reduced to 50)
        ideal_values = {}
        for ideal_col in ideal_columns:
            # Use .values for faster extraction than .to_numpy()
            ideal_values[ideal_col] = ideal_df[ideal_col].values.astype(float, copy=False)

        # OPTIMIZATION: Stack ideal values into single numpy array for NUMBA JIT compilation
        # Shape: (50 ideals, 400 samples) - enables vectorized processing
        ideal_values_stacked = np.stack([ideal_values[col] for col in ideal_columns], axis=0)

        # Pre-compute all training values as well
        training_columns = [c for c in training_df.columns if c != 'x']
        training_values = {col: training_df[col].values.astype(float, copy=False) 
                          for col in training_columns}

        for train_col in training_columns:
            train_vals = training_values[train_col]
            
            # OPTIMIZATION: Use NUMBA JIT-compiled exhaustive search
            # Searches all 50 ideal functions in compiled native code
            best_idx, sum_sq, max_dev = find_best_ideal_by_column(train_vals, ideal_values_stacked)
            
            best = SelectionResult(
                ideal_index=int(best_idx),
                sum_sq=float(sum_sq),
                max_dev=float(max_dev)
            )
            
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
