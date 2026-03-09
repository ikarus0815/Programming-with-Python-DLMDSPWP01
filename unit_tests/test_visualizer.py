import pandas as pd

from visualization import Visualizer


def test_color_mapping_handles_nan():
    # create mapping df with float and NaN ideal_func
    df = pd.DataFrame({'x':[0,1,2],'y_test':[0,1,2],'ideal_func':[1.0, float('nan'), 3]})
    # simulate color mapping used by plot_test_mappings
    colors = ['red','green','blue','orange','purple']
    # suppose only ideal indices 1 and 3 were selected, giving colors red & green
    color_map = {1: colors[0], 3: colors[1]}
    def pick_color(i):
        if i is None or (isinstance(i, float) and pd.isna(i)):
            return 'black'
        return color_map.get(int(i), 'black')
    mapped = df['ideal_func'].map(pick_color)
    assert mapped.tolist() == ['red', 'black', 'green']

def test_test_plot_circle_marker(tmp_path):
    # ensure plot_test_mappings uses circle glyphs
    training = pd.DataFrame({'x':[0,1], 'y1':[0,1], 'y2':[1,2], 'y3':[2,3], 'y4':[3,4]})
    ideal = training.copy()
    from function_searcher import SelectionResult
    selections = {f'y{i}': SelectionResult(ideal_index=i, sum_sq=0, max_dev=0)
                  for i in range(1,5)}
    from visualization import Visualizer
    mapping = pd.DataFrame({'x':[0], 'y_test':[0], 'delta_y':[0], 'ideal_func':[1]})
    fig = Visualizer.plot_test_mappings(training, mapping, ideal, selections,
                                       output_html=str(tmp_path / "out2.html"))
    # verify one of the renderers is a circle glyph
    from bokeh.models import Scatter
    glyphs = [r.glyph for r in fig.renderers if hasattr(r,'glyph')]
    assert any(isinstance(g, Scatter) and getattr(g, 'marker', None) == 'circle' for g in glyphs)


def test_plot_training_color_consistency(tmp_path):
    # construct a tiny training+ideal dataset
    training = pd.DataFrame({'x':[0,1], 'y1':[0,1], 'y2':[1,2], 'y3':[2,3], 'y4':[3,4]})
    ideal = pd.DataFrame({'x':[0,1], 'y1':[0,1], 'y2':[1,2], 'y3':[2,3], 'y4':[3,4]})
    # pretend each training column matched to same-index ideal
    from function_searcher import SelectionResult
    selections = {f'y{i}': SelectionResult(ideal_index=i, sum_sq=0, max_dev=0)
                  for i in range(1,5)}
    # call plot_training and inspect returned figure
    from visualization import Visualizer
    fig = Visualizer.plot_training(training, ideal, selections,
                                   output_html=str(tmp_path / "out.html"))
    # find line renderers and check their colors
    line_colors = [r.glyph.line_color for r in fig.renderers if hasattr(r, 'glyph')]
    # should contain at least 8 lines (4 training + 4 ideal)
    assert len(line_colors) >= 8
    # each color used for a training/ideal pair so should appear an even
    # number of times and there should be at least as many distinct colors
    # as training columns
    from collections import Counter
    counts = Counter(line_colors)
    assert all(cnt % 2 == 0 for cnt in counts.values())
    assert len(counts) >= 4
