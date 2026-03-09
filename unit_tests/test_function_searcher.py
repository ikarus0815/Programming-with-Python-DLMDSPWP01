import pandas as pd
import numpy as np
import pytest

from function_searcher import FunctionSearcher, SelectionError


def test_select_ideal_functions_basic():
    # simple training and ideal where ideal1 == training1 and ideal2 == training2
    training = pd.DataFrame({
        'x': [0,1,2],
        'y1': [0,1,2],
        'y2': [1,2,3],
        'y3': [2,3,4],
        'y4': [3,4,5],
    })
    ideal = pd.DataFrame({
        'x': [0,1,2],
        'y1': [0,1,2],
        'y2': [1,2,3],
    })
    fs = FunctionSearcher()
    res = fs.select_ideal_functions(training, ideal)
    # expect y1->y1, y2->y2, ... but only two ideals exist, so y3/y4 will choose closest
    assert res['y1'].ideal_index == 1
    assert res['y2'].ideal_index == 2
    assert res['y1'].sum_sq == 0
    assert res['y2'].sum_sq == 0


def test_mismatched_x_raises():
    training = pd.DataFrame({'x':[0,1],'y1':[0,1],'y2':[0,1],'y3':[0,1],'y4':[0,1]})
    ideal = pd.DataFrame({'x':[0,2], 'y1':[0,0]})
    fs = FunctionSearcher()
    with pytest.raises(SelectionError):
        fs.select_ideal_functions(training, ideal)
