import pandas as pd
import pytest

from loader import TrainingLoader, IdealLoader, TestLoader, LoaderError
from pathlib import Path


def make_temp_csv(tmp_path, content: str) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(content)
    return p


def test_training_loader_success(tmp_path):
    csv = "x,y1,y2,y3,y4\n1,2,3,4,5\n2,3,4,5,6\n"
    path = make_temp_csv(tmp_path, csv)
    loader = TrainingLoader()
    df = loader.load(path)
    assert list(df.columns) == ["x","y1","y2","y3","y4"]
    assert df.shape == (2,5)


def test_training_loader_bad_format(tmp_path):
    csv = "x,y1\n1,2\n"
    path = make_temp_csv(tmp_path, csv)
    loader = TrainingLoader()
    with pytest.raises(LoaderError):
        loader.load(path)


def test_ideal_loader_minimum(tmp_path):
    csv = "x,y1\n0,0\n1,1\n"
    path = make_temp_csv(tmp_path, csv)
    loader = IdealLoader()
    df = loader.load(path)
    assert list(df.columns) == ["x","y1"]


def test_test_loader_wrong_columns(tmp_path):
    csv = "x,y1,y2\n1,2,3\n"
    path = make_temp_csv(tmp_path, csv)
    loader = TestLoader()
    with pytest.raises(LoaderError):
        loader.load(path)
