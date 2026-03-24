"""Database management layer using SQLAlchemy.

Defines ORM models for training data, ideal functions and test mappings,
plus a convenience manager class that handles creation and bulk loading of
pandas DataFrames.
"""





from pathlib import Path


from sqlalchemy import Column, Float, Integer, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

import pandas as pd


Base = declarative_base()


class DatabaseError(Exception):
    """Raised for problems interacting with the SQLite database."""


class TrainingRow(Base):
    __tablename__ = "training"

    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False, index=True)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)


class IdealRow(Base):
    __tablename__ = "ideal"

    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False, index=True)


class TestMapping(Base):
    __tablename__ = "test_mapping"

    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False, index=True)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_index = Column(Integer)


class DatabaseManager:
    """Simple wrapper around SQLAlchemy engine/session for the project."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all required tables in the database."""
        try:
            Base.metadata.create_all(self.engine)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Failed to create tables: {exc}") from exc

    def load_training(self, df: pd.DataFrame) -> None:
        """Insert training data from a DataFrame into the training table."""
        try:
            df.to_sql("training", self.engine, if_exists="replace", index=False)
        except Exception as exc:
            raise DatabaseError(f"Unable to load training data: {exc}") from exc

    def load_ideal(self, df: pd.DataFrame) -> None:
        """Insert ideal functions data from a DataFrame into the ideal table."""
        try:
            df.to_sql("ideal", self.engine, if_exists="replace", index=False)
        except Exception as exc:
            raise DatabaseError(f"Unable to load ideal data: {exc}") from exc

    def store_test_results(self, df: pd.DataFrame) -> None:
        """Write the mapping results DataFrame into the test_mapping table."""
        try:
            df.to_sql("test_mapping", self.engine, if_exists="replace", index=False)
        except Exception as exc:
            raise DatabaseError(f"Unable to store test results: {exc}") from exc

    def query(self, stmt):
        """Execute a raw statement and return results (for ad‑hoc use)."""
        with self.Session() as session:
            return session.execute(stmt).fetchall()
