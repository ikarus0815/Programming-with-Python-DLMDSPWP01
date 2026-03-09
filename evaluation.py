"""Evaluation and visualization of test data against chosen ideal functions."""



import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool


class EvaluationError(Exception):
    """Raised during test data evaluation process."""


class Evaluator:
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
        # build quick lookup of ideal values by column name
        ideal_lookup = {col: self.ideal_df[col].to_numpy(dtype=float) for col in self.ideal_df.columns if col != 'x'}
        x_values = self.ideal_df['x'].to_numpy(dtype=float)

        for _, row in test_df.iterrows():
            x_val = float(row['x'])
            y_val = float(row['y'])
            # find matching index in x_values
            try:
                idx = int((x_values == x_val).argmax())
            except Exception:
                # x not found; skip or raise
                continue

            best_fit = None
            best_delta = float('inf')
            chosen_index = None

            for train_col, sel in self.selection_results.items():
                threshold = sel.max_dev * 2**0.5
                # sel.ideal_index should be an integer, but cast defensively
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


class Visualizer:
    """Creates Bokeh plots for training data, ideal curves and test mappings."""

    @staticmethod
    def plot_training(
        training_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selections: dict[str, "SelectionResult"],
        output_html: str = "training.html",
    ) -> "bokeh.plotting.Figure":
        """Plot training curves together with their matching ideal functions.

        Training and its corresponding ideal curve share the same color; the
        ideal curves are drawn dashed.  A legend is automatically created.
        The plot is written to ``output_html`` and the figure object is
        returned for programmatic inspection (useful in tests).
        """
        p = figure(title="Training vs Ideal",
                   x_axis_label="x", y_axis_label="y")
        # choose a palette that can accommodate up to four functions
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        # map training columns to colors (skip 'x')
        train_cols = [c for c in training_df.columns if c != 'x']
        col_color = {col: colors[i % len(colors)] for i, col in enumerate(train_cols)}

        # plot training curves
        for col in train_cols:
            p.line(training_df['x'], training_df[col],
                   legend_label=f"train_{col}", line_color=col_color[col])

        # plot selected ideal curves using same color but dashed
        for train_col, sel in selections.items():
            col_idx = int(sel.ideal_index)
            ideal_col = ideal_df.columns[col_idx]
            color = col_color.get(train_col, 'black')
            p.line(ideal_df['x'], ideal_df[ideal_col],
                   legend_label=f"ideal_{ideal_col}",
                   line_color=color, line_dash='dashed')

        hover = HoverTool(tooltips=[("x", "$x"), ("y", "$y")])
        p.add_tools(hover)
        p.legend.click_policy = "hide"
        output_file(output_html)
        save(p)
        return p

    @staticmethod
    def plot_test_mappings(
        test_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selections: dict[str, "SelectionResult"],
        output_html: str = "test_mappings.html",
    ) -> "bokeh.plotting.Figure":
        """Plot test points along with the chosen ideal functions.

        Test points are coloured by their assigned ideal function; the four
        selected ideal curves are drawn as solid lines using the same colour
        palette used in ``plot_training``.  A legend shows which colour
        corresponds to which ideal index.
        """
        p = figure(title="Test Mappings", x_axis_label="x", y_axis_label="y")
        # color palette reused from training plot
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        # derive a consistent color for each selected ideal index
        ideal_indices = [int(sel.ideal_index) for sel in selections.values()]
        color_map: dict[int, str] = {}
        for idx, ideal_index in enumerate(ideal_indices):
            color_map[ideal_index] = colors[idx % len(colors)]

        # plot ideal functions (only those chosen)
        for train_col, sel in selections.items():
            ideal_index = int(sel.ideal_index)
            ideal_col = ideal_df.columns[ideal_index]
            color = color_map.get(ideal_index, 'black')
            p.line(ideal_df['x'], ideal_df[ideal_col],
                   line_color=color, line_dash='dashed',
                   legend_label=f"ideal_{ideal_col}")

        # plot test points coloured by ideal index assignment
        map_df = mapping_df.copy()
        def pick_color(i):
            if i is None or (isinstance(i, float) and pd.isna(i)):
                return 'black'
            return color_map.get(int(i), 'black')
        map_df['color'] = map_df['ideal_func'].map(pick_color)
        src = ColumnDataSource(map_df)
        # draw points as circles; scatter with marker ensures future compatibility
        p.scatter('x', 'y_test', color='color', size=6, marker='circle', source=src)

        hover = HoverTool(tooltips=[("x", "@x"), ("y", "@y_test"), ("ideal", "@ideal_func")])
        p.add_tools(hover)
        p.legend.click_policy = "hide"
        output_file(output_html)
        save(p)
        return p
