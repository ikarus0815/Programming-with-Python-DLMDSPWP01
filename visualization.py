
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool


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
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        train_cols = [c for c in training_df.columns if c != 'x']
        col_color = {col: colors[i % len(colors)] for i, col in enumerate(train_cols)}

        for col in train_cols:
            p.line(training_df['x'], training_df[col],
                   legend_label=f"train_{col}", line_color=col_color[col])

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
        colors = ['red', 'green', 'blue', 'orange', 'purple']

        ideal_indices = [int(sel.ideal_index) for sel in selections.values()]
        color_map: dict[int, str] = {}
        for idx, ideal_index in enumerate(ideal_indices):
            color_map[ideal_index] = colors[idx % len(colors)]

        for train_col, sel in selections.items():
            ideal_index = int(sel.ideal_index)
            ideal_col = ideal_df.columns[ideal_index]
            color = color_map.get(ideal_index, 'black')
            p.line(ideal_df['x'], ideal_df[ideal_col],
                   line_color=color, line_dash='dashed',
                   legend_label=f"ideal_{ideal_col}")

        map_df = mapping_df.copy()
        def pick_color(i):
            if i is None or (isinstance(i, float) and pd.isna(i)):
                return 'black'
            return color_map.get(int(i), 'black')
        map_df['color'] = map_df['ideal_func'].map(pick_color)
        src = ColumnDataSource(map_df)
        p.scatter('x', 'y_test', color='color', size=6, marker='circle', source=src)

        hover = HoverTool(tooltips=[("x", "@x"), ("y", "@y_test"), ("ideal", "@ideal_func")])
        p.add_tools(hover)
        p.legend.click_policy = "hide"
        output_file(output_html)
        save(p)
        return p