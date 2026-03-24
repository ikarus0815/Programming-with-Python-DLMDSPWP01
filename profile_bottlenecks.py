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

USE_WARMUP = True


def print_stats_with_precision(ps, limit=20, decimals=6):
    """Print pstats with custom decimal precision."""
    print(f"{'ncalls':<10} {'tottime':<12} {'percall':<12} {'cumtime':<12} {'percall':<12} {'filename:lineno(function)':<50}")
    print("-" * 120)
    
    for func, stats in sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True)[:limit]:
        ncalls = stats[0]
        tottime = stats[2]
        cumtime = stats[3]
        percall_tot = tottime / ncalls if ncalls > 0 else 0
        percall_cum = cumtime / ncalls if ncalls > 0 else 0
        
        print(f"{ncalls:<10} {tottime:<12.{decimals}f} {percall_tot:<12.{decimals}f} {cumtime:<12.{decimals}f} {percall_cum:<12.{decimals}f} {str(func):<50}")


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
    print_stats_with_precision(ps, limit=20, decimals=4)


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
    print_stats_with_precision(ps, limit=25, decimals=4)


if __name__ == "__main__":


    profile_function_selection(iterations=5)
    profile_test_mapping(iterations=5)

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)