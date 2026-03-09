"""Data loading utilities for the project.

This module defines an abstract base class for CSV loaders and concrete
implementations for training, ideal and test datasets.  Loaders return
pandas DataFrames and perform basic validation on the contents.
"""



import abc
import pandas as pd
from pathlib import Path


class LoaderError(Exception):
    """Raised when a dataset cannot be loaded or is malformed."""


class BaseLoader(abc.ABC):
    """Abstract base class for CSV dataset loaders."""

    @abc.abstractmethod
    def load(self, path: str | Path) -> pd.DataFrame:
        """Read the data from ``path`` and return a validated DataFrame.

        Parameters
        ----------
        path : str | Path
            Location of the CSV file to read.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded data.

        Raises
        ------
        LoaderError
            If the file cannot be read or the contents do not match the
            expected format.
        """

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Ensure ``df`` has at least two columns (x and y) with numeric data.

        Subclasses may override this method to enforce a more specific
        structure.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Raises
        ------
        LoaderError
            If the DataFrame does not meet the expected requirements.
        """
        if df.shape[1] < 2:
            raise LoaderError("DataFrame must have at least two columns")
        # check for numeric dtype
        if not all(pd.api.types.is_numeric_dtype(dt) for dt in df.dtypes):
            raise LoaderError("All columns must be numeric")


class TrainingLoader(BaseLoader):
    """Loader for the training dataset (one CSV containing x and four y columns)."""

    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise LoaderError(f"Training file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise LoaderError(f"Failed to read training CSV: {exc}") from exc

        self._validate_dataframe(df)
        # expecting exactly 5 columns: x, y1..y4
        if df.shape[1] != 5:
            raise LoaderError("Training CSV must have exactly 5 columns: x,y1..y4")
        return df


class IdealLoader(BaseLoader):
    """Loader for the ideal-functions dataset (x plus 50 y columns)."""

    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise LoaderError(f"Ideal functions file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise LoaderError(f"Failed to read ideal CSV: {exc}") from exc

        self._validate_dataframe(df)
        if df.shape[1] < 2:
            raise LoaderError("Ideal CSV must contain at least x and one function")
        return df


class TestLoader(BaseLoader):
    """Loader for the test dataset (two-column x,y file)."""

    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise LoaderError(f"Test file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise LoaderError(f"Failed to read test CSV: {exc}") from exc

        self._validate_dataframe(df)
        if df.shape[1] != 2:
            raise LoaderError("Test CSV must have exactly two columns: x,y")
        return df
