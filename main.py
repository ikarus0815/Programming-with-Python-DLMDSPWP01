"""Simple runner for the function-fitting program using fixed paths.

Usage:
    python main.py  # Run the full pipeline
"""

import sys

from loader import TrainingLoader, IdealLoader, TestLoader, LoaderError
from database_manager import DatabaseManager, DatabaseError
from function_searcher import FunctionSearcher, SelectionError
from mapping_test import Mapping, MappingError
from visualization import Visualizer
from performance_tracker import PerformanceTracker

def main() -> None:
    # hard-coded paths – adjust as needed
    training_path = "data/train.csv"
    ideal_path = "data/ideal.csv"
    test_path = "data/test.csv"
    output_db = "results.db"

    # Initialize performance tracker
    tracker = PerformanceTracker()

    try:
        # Phase 1: Data Loading
        tracker.start("data_loading")
        t_loader = TrainingLoader()
        training_df = t_loader.load(training_path)

        i_loader = IdealLoader()
        ideal_df = i_loader.load(ideal_path)

        test_loader = TestLoader()
        test_df = test_loader.load(test_path)
        tracker.end("data_loading")

        # Phase 2: Database Initialization
        tracker.start("database_init")
        db = DatabaseManager(output_db)
        db.create_tables()
        tracker.end("database_init")

        # Phase 3: Data Persistence
        tracker.start("database_load")
        db.load_training(training_df)
        db.load_ideal(ideal_df)
        tracker.end("database_load")

        # Phase 4: Function Selection (CPU-intensive)
        tracker.start("function_selection")
        searcher = FunctionSearcher()
        selections = searcher.select_ideal_functions(training_df, ideal_df)
        tracker.end("function_selection")

        # Phase 5: Test Point Mapping (CPU-intensive)
        tracker.start("test_mapping")
        evaluator = Mapping(ideal_df, selections)
        mapping_df = evaluator.map_test_points(test_df)
        tracker.end("test_mapping")

        # Phase 6: Result Storage
        tracker.start("result_storage")
        evaluator.save_mapping(mapping_df, db)
        tracker.end("result_storage")

        # Phase 7: Visualization
        tracker.start("visualization")
        Visualizer.plot_training(training_df, ideal_df, selections)
        Visualizer.plot_test_mappings(test_df, mapping_df, ideal_df, selections)
        tracker.end("visualization")

        # Print performance summary
        tracker.print_summary()

        print("\nProcessing complete.")
        for train_col, sel in selections.items():
            print(f"{train_col} -> ideal index {sel.ideal_index} (sum_sq={sel.sum_sq:.2f})")
        print(f"Test results written to {output_db}")

    except LoaderError as err:
        tracker.end(tracker.current_phase) if tracker.current_phase else None
        print(f"Data loading failed: {err}", file=sys.stderr)
        sys.exit(1)
    except DatabaseError as err:
        tracker.end(tracker.current_phase) if tracker.current_phase else None
        print(f"Database error: {err}", file=sys.stderr)
        sys.exit(1)
    except SelectionError as err:
        tracker.end(tracker.current_phase) if tracker.current_phase else None
        print(f"Selection error: {err}", file=sys.stderr)
        sys.exit(1)
    except MappingError as err:
        tracker.end(tracker.current_phase) if tracker.current_phase else None
        print(f"Mapping error: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        tracker.end(tracker.current_phase) if tracker.current_phase else None
        import traceback
        print("Unexpected error:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()