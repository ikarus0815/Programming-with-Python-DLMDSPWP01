"""
Benchmark Mode: Isolated performance measurement for core algorithms.

Executes function_searcher and mapping_test 25 times with statistical analysis
to measure performance independently of I/O overhead (visualization, database).

This module provides clean, reproducible performance metrics by:
- Pre-loading all data once (no I/O during benchmark)
- Running warmup iterations (CPU cache population, JIT optimization)
- Collecting 25 benchmark runs with garbage collection between iterations
- Computing statistical metrics (median, mean, stdev, p95) to reduce noise
"""

import gc
import time
import statistics
import json
from pathlib import Path
from typing import Dict, Any

from loader import TrainingLoader, IdealLoader, TestLoader
from function_searcher import FunctionSearcher
from mapping_test import Mapping


class BenchmarkRunner:
    """Measures core algorithm performance with minimal I/O interference."""

    def __init__(self, training_path: str = "data/train.csv",
                 ideal_path: str = "data/ideal.csv",
                 test_path: str = "data/test.csv"):
        """
        Pre-load all data once to avoid I/O noise during benchmarking.

        Parameters
        ----------
        training_path : str
            Path to training data CSV
        ideal_path : str
            Path to ideal functions CSV
        test_path : str
            Path to test data CSV
        """
        print("Loading benchmark data...")
        self.training_df = TrainingLoader().load(training_path)
        self.ideal_df = IdealLoader().load(ideal_path)
        self.test_df = TestLoader().load(test_path)
        print("✓ Data loaded\n")

    def benchmark_function_selection(self, iterations: int = 25, warmup: int = 1) -> Dict[str, Any]:
        """
        Benchmark the function selection algorithm over multiple iterations.

        Measures FunctionSearcher.select_ideal_functions() performance by:
        1. Running warmup iterations (not counted)
        2. Executing 25 benchmark iterations with garbage collection between
        3. Collecting wall-clock timing data
        4. Computing statistics

        Parameters
        ----------
        iterations : int
            Number of benchmark iterations (default: 25)
        warmup : int
            Number of warmup iterations (default: 1)

        Returns
        -------
        Dict[str, Any]
            Statistics including: min, max, mean, median, stdev, p95, all timings
        """
        def run_selection():
            searcher = FunctionSearcher()
            return searcher.select_ideal_functions(self.training_df, self.ideal_df)

        # Warmup iterations (not counted)
        print(f"Warmup ({warmup} iteration)...", end="", flush=True)
        for _ in range(warmup):
            run_selection()
        print(" ✓")

        # Benchmark iterations
        print(f"Benchmarking ({iterations} iterations):", end=" ", flush=True)
        times = []
        for i in range(iterations):
            if i > 0 and i % 5 == 0:
                print(f"{i}", end=" ", flush=True)

            # Clean garbage between runs
            gc.collect()

            # Measure execution time
            start = time.perf_counter()
            result = run_selection()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"{iterations} ✓")

        return {
            'phase': 'function_selection',
            'iterations': len(times),
            'min': min(times),
            'max': max(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': sorted(times)[int(0.95 * len(times))],
            'times': times,
        }

    def benchmark_test_mapping(self, iterations: int = 25, warmup: int = 1) -> Dict[str, Any]:
        """
        Benchmark the test point mapping algorithm over multiple iterations.

        Measures Mapping.map_test_points() performance. Pre-computes selections
        once to focus measurement on mapping algorithm only.

        Parameters
        ----------
        iterations : int
            Number of benchmark iterations (default: 25)
        warmup : int
            Number of warmup iterations (default: 1)

        Returns
        -------
        Dict[str, Any]
            Statistics including: min, max, mean, median, stdev, p95, all timings
        """
        # Pre-compute selections once (not part of benchmark)
        print("Preparing selections...", end="", flush=True)
        searcher = FunctionSearcher()
        selections = searcher.select_ideal_functions(self.training_df, self.ideal_df)
        print(" ✓")

        def run_mapping():
            evaluator = Mapping(self.ideal_df, selections)
            return evaluator.map_test_points(self.test_df)

        # Warmup iterations (not counted)
        print(f"Warmup ({warmup} iteration)...", end="", flush=True)
        for _ in range(warmup):
            run_mapping()
        print(" ✓")

        # Benchmark iterations
        print(f"Benchmarking ({iterations} iterations):", end=" ", flush=True)
        times = []
        for i in range(iterations):
            if i > 0 and i % 5 == 0:
                print(f"{i}", end=" ", flush=True)

            # Clean garbage between runs
            gc.collect()

            # Measure execution time
            start = time.perf_counter()
            result = run_mapping()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        print(f"{iterations} ✓")

        return {
            'phase': 'test_mapping',
            'iterations': len(times),
            'min': min(times),
            'max': max(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'p95': sorted(times)[int(0.95 * len(times))],
            'times': times,
        }

    @staticmethod
    def print_stats(stats_dict: Dict[str, Any]) -> None:
        """
        Pretty-print benchmark statistics for a single phase.

        Parameters
        ----------
        stats_dict : Dict[str, Any]
            Statistics dictionary from benchmark_* methods
        """
        phase = stats_dict['phase']
        iterations = stats_dict['iterations']

        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS: {phase.upper()} ({iterations} iterations)")
        print(f"{'='*70}")

        # Convert to milliseconds for readability
        min_ms = stats_dict['min'] * 1000
        max_ms = stats_dict['max'] * 1000
        mean_ms = stats_dict['mean'] * 1000
        median_ms = stats_dict['median'] * 1000
        stdev_ms = stats_dict['stdev'] * 1000
        p95_ms = stats_dict['p95'] * 1000
        range_ms = (stats_dict['max'] - stats_dict['min']) * 1000

        # Calculate coefficient of variation (CV) to assess noise level
        cv = (stats_dict['stdev'] / stats_dict['mean'] * 100) if stats_dict['mean'] > 0 else 0

        print(f"  Median:      {median_ms:8.3f} ms (most representative value)")
        print(f"  Mean:        {mean_ms:8.3f} ms (±{stdev_ms:6.3f} ms)")
        print(f"  Min / Max:   {min_ms:8.3f} / {max_ms:8.3f} ms")
        print(f"  P95:         {p95_ms:8.3f} ms (95th percentile)")
        print(f"  Range:       {range_ms:8.3f} ms ({cv:5.2f}% coefficient of variation)")
        print()

    @staticmethod
    def export_results(results: Dict[str, Dict], output_file: str = "benchmark_results.json") -> None:
        """
        Export benchmark results to JSON file.

        Parameters
        ----------
        results : Dict[str, Dict]
            Benchmark results from run_full_benchmark()
        output_file : str
            Output filename (default: benchmark_results.json)
        """
        # Remove detailed timing arrays to reduce file size
        export_data = {}
        for phase, stats in results.items():
            export_data[phase] = {k: v for k, v in stats.items() if k != 'times'}

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Results exported to {output_file}")

    def run_full_benchmark(self, iterations: int = 25, warmup: int = 1) -> Dict[str, Dict]:
        """
        Run complete benchmark suite for both algorithms.

        Parameters
        ----------
        iterations : int
            Number of benchmark iterations per algorithm
        warmup : int
            Number of warmup iterations

        Returns
        -------
        Dict[str, Dict]
            Complete benchmark results for all phases
        """
        print("\n" + "="*70)
        print("STARTING BENCHMARK SUITE (25 iterations each)")
        print("="*70 + "\n")

        results = {}

        # Benchmark 1: Function Selection
        print("1️⃣  FUNCTION SELECTION")
        print("-" * 70)
        stats_fs = self.benchmark_function_selection(iterations=iterations, warmup=warmup)
        self.print_stats(stats_fs)
        results['function_selection'] = stats_fs

        # Benchmark 2: Test Mapping
        print("2️⃣  TEST MAPPING")
        print("-" * 70)
        stats_tm = self.benchmark_test_mapping(iterations=iterations, warmup=warmup)
        self.print_stats(stats_tm)
        results['test_mapping'] = stats_tm

        # Summary
        print("="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print(f"\nSummary:")
        print(f"  Function Selection: {stats_fs['median']*1000:.3f}ms (median)")
        print(f"  Test Mapping:       {stats_tm['median']*1000:.3f}ms (median)")
        total_median = (stats_fs['median'] + stats_tm['median']) * 1000
        print(f"  Total (algorithm):  {total_median:.3f}ms")

        return results


def main():
    """Run benchmark suite with 25 iterations per algorithm."""
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark(iterations=25, warmup=1)
    runner.export_results(results)


if __name__ == "__main__":
    main()
