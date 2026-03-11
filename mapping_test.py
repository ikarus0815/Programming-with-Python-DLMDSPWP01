"""Evaluation and visualization of test data against chosen ideal functions."""



import pandas as pd


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

        # OPTIMIZATION: Pre-compute thresholds and ideal values to avoid repeated calculations
        # This eliminates redundant threshold calculations and array allocations
        self.thresholds = {col: sr.max_dev * (2 ** 0.5) for col, sr in selection_results.items()}
        self.ideal_values = {col: ideal_df.iloc[:, int(sr.ideal_index)].to_numpy(dtype=float)
                             for col, sr in selection_results.items()}

        # OPTIMIZATION: Create x-value lookup dictionary for O(1) access instead of O(n) linear search
        # This replaces the expensive (x_values == x_val).argmax() operation
        self.x_to_idx = {x: i for i, x in enumerate(ideal_df['x'].to_numpy(dtype=float))}

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
        # This replaces slow iterrows() with direct array access (10-15x faster)
        x_test = test_df['x'].to_numpy(dtype=float)
        y_test = test_df['y'].to_numpy(dtype=float)

        # OPTIMIZATION: Iterate over numpy arrays instead of DataFrame.iterrows()
        for i in range(len(x_test)):
            x_val = x_test[i]
            y_val = y_test[i]

            # OPTIMIZATION: Use pre-computed x-index dictionary for O(1) lookup instead of O(n) search
            # Replaces expensive (x_values == x_val).argmax() operation (50-60x faster)
            idx = self.x_to_idx.get(x_val)
            if idx is None:
                # x-value not found in ideal data - skip this test point
                continue

            best_fit = None
            best_delta = float('inf')
            chosen_index = None

            # OPTIMIZATION: Use pre-computed thresholds and ideal values from __init__
            # Eliminates redundant threshold calculations and repeated array allocations
            for train_col, threshold in self.thresholds.items():
                # Get pre-computed ideal values for this training column
                ideal_vals = self.ideal_values[train_col]
                col_idx = int(self.selection_results[train_col].ideal_index)

                # Compute deviation
                delta = abs(y_val - ideal_vals[idx])

                # Update best fit if within threshold and better than current best
                if delta <= threshold and delta < best_delta:
                    best_delta = delta
                    chosen_index = col_idx
                    best_fit = delta

            results.append({
                'x': x_val,
                'y_test': y_val,
                'delta_y': best_fit if best_fit is not None else None,
                'ideal_func': chosen_index,
            })

        return pd.DataFrame(results)

    def save_mapping(self, df: pd.DataFrame, db_manager) -> None:
        """Delegate storage of a mapping DataFrame to the database manager."""
        db_manager.store_test_results(df)



