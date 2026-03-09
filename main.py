"""Simple runner for the function-fitting program using fixed paths.
"""

import sys

from loader import TrainingLoader, IdealLoader, TestLoader, LoaderError
from database_manager import DatabaseManager, DatabaseError
from function_searcher import FunctionSearcher, SelectionError
from mapping_test import Mapping, MappingError
from visualization import Visualizer


def main() -> None:
    # hard-coded paths – adjust as needed
    training_path = "data/train.csv"
    ideal_path = "data/ideal.csv"
    test_path = "data/test.csv"
    output_db = "results.db"

    try:
        t_loader = TrainingLoader()
        training_df = t_loader.load(training_path)

        i_loader = IdealLoader()
        ideal_df = i_loader.load(ideal_path)

        test_loader = TestLoader()
        test_df = test_loader.load(test_path)

        db = DatabaseManager(output_db)
        db.create_tables()
        db.load_training(training_df)
        db.load_ideal(ideal_df)

        searcher = FunctionSearcher()
        selections = searcher.select_ideal_functions(training_df, ideal_df)

        evaluator = Mapping(ideal_df, selections)
        mapping_df = evaluator.map_test_points(test_df)
        evaluator.save_mapping(mapping_df, db)

        Visualizer.plot_training(training_df, ideal_df, selections)
        Visualizer.plot_test_mappings(test_df, mapping_df, ideal_df, selections)

        print("Processing complete.")
        for train_col, sel in selections.items():
            print(f"{train_col} -> ideal index {sel.ideal_index} (sum_sq={sel.sum_sq:.2f})")
        print(f"Test results written to {output_db}")

    except LoaderError as err:
        print(f"Data loading failed: {err}", file=sys.stderr)
        sys.exit(1)
    except DatabaseError as err:
        print(f"Database error: {err}", file=sys.stderr)
        sys.exit(1)
    except SelectionError as err:
        print(f"Selection error: {err}", file=sys.stderr)
        sys.exit(1)
    except MappingError as err:
        print(f"Mapping error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        import traceback
        print("Unexpected error:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
