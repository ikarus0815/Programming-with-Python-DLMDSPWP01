"""Evaluation and visualization of test data against chosen ideal functions."""



import pandas as pd
import numpy as np
from numba_kernels import find_best_test_mapping, batch_find_best_mapping, batch_find_x_indices, NUMBA_AVAILABLE


class MappingError(Exception):
    """Raised during test data evaluation process."""


class Mapping:
    """Maps test points to selected ideal functions using given thresholds."""

    def __init__(
        self,
        ideal_df: pd.DataFrame,
        selection_results: dict[str, "SelectionResult"],
    ) -> None:
        self.ideal_df = ideal_df
        self.selection_results = selection_results

        # OPTIMIZATION: Extract columns to numpy arrays ONCE for entire lifetime
        self.x_values = ideal_df['x'].to_numpy(dtype=float)
        
        # Pre-compute thresholds and ideal values in numpy format
        self.thresholds = {col: sr.max_dev * (2 ** 0.5) for col, sr in selection_results.items()}
        
        # OPTIMIZATION: Cache ideal functions as numpy arrays indexed by column name (efficient)
        self.ideal_values_array = {}
        for train_col, sr in selection_results.items():
            col_idx = int(sr.ideal_index)
            # Direct numpy extraction is faster than .iloc[:, col_idx].to_numpy()
            self.ideal_values_array[train_col] = ideal_df.iloc[:, col_idx].values.astype(float, copy=False)

        # OPTIMIZATION: Stack ideal values for NUMBA JIT compilation
        train_cols = [col for col in selection_results.keys()]
        self.ideal_values_stacked = np.stack(
            [self.ideal_values_array[col] for col in train_cols],
            axis=0
        )
        self.train_cols = train_cols

        # Pre-compute thresholds as numpy array for NUMBA (matches stacked order)
        self.thresholds_array = np.array(
            [self.thresholds[col] for col in train_cols],
            dtype=np.float64
        )

        # OPTIMIZATION: Create x-value lookup dictionary for O(1) access instead of O(n) linear search
        self.x_to_idx = {x: i for i, x in enumerate(self.x_values)}

    def map_test_points(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Map each row of ``test_df`` to one of the ideal functions.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame with columns ['x','y'] representing test measurements.

        Returns
        -------
        pd.DataFrame
            DataFrame containing columns ``x``, ``y_test``, ``delta_y`` and
            ``ideal_func`` (the index of the chosen ideal function or ``None``).
        """
        results: list[dict] = []

        # OPTIMIZATION: Convert test data to numpy arrays for faster iteration
        x_test = test_df['x'].to_numpy(dtype=float)
        y_test = test_df['y'].to_numpy(dtype=float)

        # OPTIMIZATION: Pre-compute x-indices for ALL test points at once (batch)
        # Maps each test x-value to its position in ideal data
        x_indices = np.array(
            [self.x_to_idx.get(x, -1) for x in x_test],
            dtype=np.int64
        )

        # OPTIMIZATION: Process ALL 100 test points in ONE NUMBA-compiled call
        # The internal 100-point loop stays in NUMBA native code (not Python!)
        deltas, col_indices = batch_find_best_mapping(
            y_test,
            self.thresholds_array,
            self.ideal_values_stacked,
            x_indices
        )

        # OPTIMIZATION: Build results array from vectorized outputs (minimal Python overhead)
        for i in range(len(x_test)):
            x_val = x_test[i]
            y_val = y_test[i]
            delta_val = deltas[i]
            col_idx = col_indices[i]

            if x_indices[i] < 0:
                continue

            chosen_index = None
            if col_idx >= 0:  
                chosen_index = int(self.selection_results[self.train_cols[int(col_idx)]].ideal_index)

            results.append({
                'x': x_val,
                'y_test': y_val,
                'delta_y': float(delta_val) if not np.isnan(delta_val) else None,
                'ideal_func': chosen_index,
            })

        return pd.DataFrame(results)

    def save_mapping(self, df: pd.DataFrame, db_manager) -> None:
        """Delegate storage of a mapping DataFrame to the database manager."""
        db_manager.store_test_results(df)



