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
        ideal_lookup = {col: self.ideal_df[col].to_numpy(dtype=float) for col in self.ideal_df.columns if col != 'x'}
        x_values = self.ideal_df['x'].to_numpy(dtype=float)

        for _, row in test_df.iterrows():
            x_val = float(row['x'])
            y_val = float(row['y'])
            try:
                idx = int((x_values == x_val).argmax())
            except Exception:
                continue

            best_fit = None
            best_delta = float('inf')
            chosen_index = None

            for train_col, sel in self.selection_results.items():
                threshold = sel.max_dev * 2**0.5
                col_idx = int(sel.ideal_index)
                ideal_vals = self.ideal_df.iloc[:, col_idx].to_numpy(dtype=float)
                delta = abs(y_val - ideal_vals[idx])
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



