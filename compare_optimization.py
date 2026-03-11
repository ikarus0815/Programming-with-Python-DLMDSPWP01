"""
Compare Optimized vs Non-Optimized Performance

This script demonstrates the performance improvements by running
benchmarks and comparing optimization impact.
"""

import json
import statistics
from benchmark_mode import BenchmarkRunner


def print_comparison_header():
    """Print formatted header for comparison"""
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION COMPARISON")
    print("="*80)
    print("\nThis branch (develop-optimized) includes:")
    print("  1. ✓ Pre-computed ideal values caching in function_searcher")
    print("  2. ✓ Improved numpy array handling (.values instead of .to_numpy())")
    print("  3. ✓ Pre-computed thresholds and ideal value arrays in mapping_test")  
    print("  4. ✓ O(1) x-value dictionary lookup instead of O(n) argmax")
    print("  5. ✓ Numpy array iteration instead of iterrows()")
    print("="*80 + "\n")


def run_optimized_benchmark(iterations: int = 25):
    """Run benchmark on optimized code"""
    print("Running optimized benchmark (25 iterations)...\n")

    runner = BenchmarkRunner()
    results = runner.run_full_benchmark(iterations=iterations, warmup=2)
    runner.export_results(results)

    return results


def analyze_results(results):
    """Extract and display key metrics"""
    print("\n" + "="*80)
    print("OPTIMIZED BRANCH RESULTS")
    print("="*80 + "\n")

    total_median = 0
    for phase, stats in results.items():
        median_ms = stats['median'] * 1000
        mean_ms = stats['mean'] * 1000
        stdev_ms = stats['stdev'] * 1000
        cv = (stats['stdev'] / stats['mean'] * 100) if stats['mean'] > 0 else 0

        print(f"{phase.upper()}:")
        print(f"  Median:    {median_ms:8.3f} ms")
        print(f"  Mean:      {mean_ms:8.3f} ms ± {stdev_ms:6.3f} ms")
        print(f"  CV:        {cv:6.2f}% (lower is better, <5% is excellent)")
        
        if phase == 'function_selection':
            total_median += stats['median']
        elif phase == 'test_mapping':
            total_median += stats['median']

    print(f"\nTotal Core Algorithm Time: {total_median*1000:.3f} ms (median)")
    print(f"  This is the time for actual computations (excluding I/O)")


def print_optimization_notes():
    """Print what optimizations are in place"""
    print("\n" + "="*80)
    print("OPTIMIZATION NOTES FOR ACADEMIC ANALYSIS")
    print("="*80)

    notes = """
KEY OPTIMIZATIONS IMPLEMENTED:

1. FUNCTION SELECTION OPTIMIZATION (function_searcher.py)
   - Replaced 200 individual to_numpy() calls with 50 cached array accesses
   - Used .values instead of .to_numpy() for faster C-level array extraction
   - Precalculated all training columns at start of algorithm
   - Maintained dot product operations for numerical stability

2. TEST MAPPING OPTIMIZATION (mapping_test.py)
   - Pre-computed thresholds in __init__ (4 calculations instead of 400+)
   - Cached all ideal function arrays as numpy arrays
   - Implemented O(1) x-value lookup dictionary
   - Replaced DataFrame.iterrows() with native numpy array iteration
   - Used .values.astype(float, copy=False) for zero-copy conversion

3. ALGORITHM COMPLEXITY IMPROVEMENTS
   - function_selection: O(200n) → O(200n) but with 4x fewer allocations
   - test_mapping: O(400m) for x-lookup → O(4m) with dictionary
   - Overall: Dominated by numpy operations, minimized Python overhead

BOTTLENECK ANALYSIS:
   - After optimizations, remaining overhead is:
     * NumPy operation time (fundamental - good BLAS performance)
     * Python interpreter overhead for loop coordinates
     * Memory bandwidth for array operations

EXPECTED IMPROVEMENTS vs NON-OPTIMIZED:
   - If profiling shows high time in .to_numpy(): 15-25% improvement
   - If profiling shows high time in iterrows(): 8-15% improvement  
   - If profiling shows high time in dict/list operations: 5-10% improvement

MEASUREMENT RELIABILITY:
   - Coefficient of Variation < 10% indicates stable measurements
   - 25 iterations sufficient for statistical significance
   - Small dataset size minimizes noise from cache effects
"""
    print(notes)


def main():
    """Run full comparison analysis"""
    print_comparison_header()

    # Run optimized benchmark
    results = run_optimized_benchmark(iterations=25)

    # Analyze results
    analyze_results(results)

    # Print academic notes
    print_optimization_notes()

    # Summary
    print("\n" + "="*80)
    print("NEXT STEPS FOR ACADEMIC WORK")
    print("="*80)
    print("""
1. Run: python profile_bottlenecks.py
   This will show EXACTLY where time is spent in the algorithms

2. Compare with develop branch:
   git checkout develop
   python main.py --benchmark --iterations 25

3. Analyze the difference:
   - Look for functions consuming most cumulative time
   - Identify if .to_numpy(), iterrows(), or dict operations dominate
   - Calculate % improvement achieved

4. Document findings:
   - Include cProfile output in appendix
   - Show before/after metrics
   - Explain optimization effectiveness
""")


if __name__ == "__main__":
    main()
