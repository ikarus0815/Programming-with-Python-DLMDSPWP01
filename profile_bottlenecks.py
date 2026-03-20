"""
Profile Bottlenecks: Identifies true performance hotspots using cProfile.

Analyzes the core algorithms to find where time is actually spent,
bypassing micro-optimization guesses.
"""

import cProfile
import pstats
import io
from loader import TrainingLoader, IdealLoader, TestLoader
from function_searcher import FunctionSearcher
from mapping_test import Mapping

# Configuration: Set to False to skip warm-up cycles
USE_WARMUP = True


def profile_function_selection(iterations: int = 5):
    """Profile FunctionSearcher.select_ideal_functions()"""
    # Load data once
    training_df = TrainingLoader().load("data/train.csv")
    ideal_df = IdealLoader().load("data/ideal.csv")

    print("\n" + "="*80)
    print("PROFILING: function_searcher.select_ideal_functions()")
    print("="*80 + "\n")

    # Warm-up cycle to exclude compilation time
    if USE_WARMUP:
        print("Running warm-up cycle...")
        searcher = FunctionSearcher()
        searcher.select_ideal_functions(training_df, ideal_df)
        print("Warm-up complete. Starting actual profiling...\n")

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(iterations):
        searcher = FunctionSearcher()
        searcher.select_ideal_functions(training_df, ideal_df)

    pr.disable()

    # Print top 20 functions by cumulative time
    ps = pstats.Stats(pr)
    ps.sort_stats('cumulative')
    ps.print_stats(20)

    # Also show by total time
    print("\n" + "="*80)
    print("Top functions by TOTAL TIME (not cumulative):")
    print("="*80 + "\n")
    ps.sort_stats('time')
    ps.print_stats(15)


def profile_test_mapping(iterations: int = 5):
    """Profile Mapping.map_test_points()"""
    # Load data once
    training_df = TrainingLoader().load("data/train.csv")
    ideal_df = IdealLoader().load("data/ideal.csv")
    test_df = TestLoader().load("data/test.csv")

    # Pre-compute selections (not part of profile)
    searcher = FunctionSearcher()
    selections = searcher.select_ideal_functions(training_df, ideal_df)

    print("\n" + "="*80)
    print("PROFILING: Mapping.map_test_points()")
    print("="*80 + "\n")

    # Warm-up cycle to exclude compilation time
    if USE_WARMUP:
        print("Running warm-up cycle...")
        evaluator = Mapping(ideal_df, selections)
        evaluator.map_test_points(test_df)
        print("Warm-up complete. Starting actual profiling...\n")

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(iterations):
        evaluator = Mapping(ideal_df, selections)
        evaluator.map_test_points(test_df)

    pr.disable()

    # Print top 25 functions by cumulative time
    ps = pstats.Stats(pr)
    ps.sort_stats('cumulative')
    ps.print_stats(25)

    # Also show by total time
    print("\n" + "="*80)
    print("Top functions by TOTAL TIME (not cumulative):")
    print("="*80 + "\n")
    ps.sort_stats('time')
    ps.print_stats(20)


if __name__ == "__main__":
    print("\nStarting profiling analysis...")
    print("This will show WHERE time is actually spent in the algorithms.\n")

    profile_function_selection(iterations=5)
    profile_test_mapping(iterations=5)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nInterpretation guide:")
    print("  - 'ncalls': Number of function calls")
    print("  - 'tottime': Total time in THIS function (excluding subfunctions)")
    print("  - 'cumtime': Total time in this function and all subfunctions")
    print("\nLook for functions with HIGH cumtime but LOW tottime - those call expensive subfunctions.")
    print("Look for functions with HIGH tottime - those are the actual bottlenecks.")
