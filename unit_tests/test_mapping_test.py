import pandas as pd
import pytest

from mapping_test import Mapping
from function_searcher import SelectionResult


def test_map_test_points_simple():
    # create ideal dataframe with x=0,1 and two ideal functions
    ideal = pd.DataFrame({
        'x': [0,1],
        'y1': [0, 0],
        'y2': [1, 1],
    })
    # pretend training chose ideal1 for first, ideal2 for others
    selections = {
        'y1': SelectionResult(ideal_index=1, sum_sq=0, max_dev=0),
        'y2': SelectionResult(ideal_index=2, sum_sq=0, max_dev=0),
    }
    evaluator = Mapping(ideal, selections)
    test_df = pd.DataFrame({'x':[0,1,0.5], 'y':[0,1,0.1]})
    mapped = evaluator.map_test_points(test_df)
    assert 'ideal_func' in mapped.columns
    # first row matches ideal1 exactly
    assert mapped.loc[0,'ideal_func'] == 1


def test_map_test_points_no_match():
    ideal = pd.DataFrame({'x':[0], 'y1':[0]})
    selections = {'y1': SelectionResult(ideal_index=1, sum_sq=0, max_dev=0)}
    evaluator = Mapping(ideal, selections)
    test_df = pd.DataFrame({'x':[0], 'y':[10]})
    mapped = evaluator.map_test_points(test_df)
    # should record None for ideal_func (no fit within threshold)
    assert mapped.loc[0,'ideal_func'] is None
