"""
OPTIMIZATION TESTING GUIDE - develop-optimized branch

Quick start for measuring and analyzing performance improvements.
"""

# Execute these commands in order to test the optimizations:

"""
STEP 1: Profile the optimized code to find bottlenecks
    python profile_bottlenecks.py
    
    This shows:
    - Where time is actually spent in the algorithms
    - Which functions call expensive subfunctions
    - The real hotspots vs. assumed bottlenecks

STEP 2: Run the optimized benchmark (25 iterations)
    python main.py --benchmark --iterations 25
    
    Output file: benchmark_results.json
    Contains:
    - Median execution time (most stable metric)
    - Mean and standard deviation  
    - Min/Max values
    - 95th percentile

STEP 3: Run comprehensive optimization analysis
    python compare_optimization.py
    
    This provides:
    - Summary of all optimizations in place
    - Detailed results with coefficient of variation
    - Notes for academic documentation

STEP 4: Compare with non-optimized (develop branch)
    git checkout develop
    python main.py --benchmark --iterations 25
    
    Compare the benchmark_results.json files:
    - develop: baseline (no optimizations)
    - develop-optimized: with all optimizations

STEP 5: Calculate performance improvement
    (develop time - develop-optimized time) / develop time * 100%
    
    Expected: 10-40% depending on which bottlenecks dominated

---

DETAILED OPTIMIZATION BREAKDOWN:

function_searcher.py CHANGES:
  OLD: ideal_df[col].to_numpy(dtype=float)  # Called 200 times per run
  NEW: Pre-computed dict with .values once  # Called 200 times from cache
  
  Benefit: Reduces array allocation overhead
  Expected gain: 5-15% if allocation was bottleneck

mapping_test.py CHANGES:  
  OLD: test_df.iterrows()                    # ~100 loop iterations
  NEW: numpy array direct access             # ~100 fast iterations
  
  Benefit: Eliminates pandas row wrapper overhead  
  Expected gain: 3-10% if loop was bottleneck
  
  OLD: (x_values == x_val).argmax()         # O(n) per test point
  NEW: self.x_to_idx.get(x_val)             # O(1) per test point
  
  Benefit: Eliminates 100 linear searches (50-100x faster lookup)
  Expected gain: 15-25% if lookup was major contributor

  OLD: Threshold calculated per iteration    
  NEW: Pre-computed in __init__
  
  Benefit: Eliminates 400 multiplication operations
  Expected gain: 1-3% (minor)

---

EXPECTED RESULTS INTERPRETATION:

If you see < 5% improvement:
  → Bottleneck was NOT in the optimized sections
  → Use cProfile output to find where time actually goes
  → Consider I/O vs computation tradeoffs

If you see 10-25% improvement:
  → Optimizations hit one major bottleneck
  → Look at cProfile to confirm which optimization helped most
  → Good for academic analysis

If you see 25%+ improvement:
  → Multiple optimizations worked together
  → Likely indicates .to_numpy() or iterrows() was major issue
  → Excellent for thesis material

---

FOR ACADEMIC DOCUMENTATION:

Include in your report:
1. cProfile output showing top functions
2. Benchmark results (median ± stdev) for both branches
3. % improvement calculation
4. Explanation of which optimization had most impact
5. Discussion of why other optimizations had limited impact
6. Coefficient of variation as measure of reproducibility
7. Notes on dataset size effects on optimization benefit

---

TROUBLESHOOTING:

Q: Why don't I see improvement despite optimizations?
A: Check profile_bottlenecks.py output - maybe bottleneck is in:
   - Database operations (not profiled in benchmark)
   - Visualization (not run in benchmark)
   - Other code not in core algorithms
   
Q: Variance too high to see improvement?
A: This is normal for small datasets
   - Increase benchmark iterations (--iterations 50)
   - Look at MEDIAN not mean (more stable)
   - Check coefficient of variation
   - Consider using py-spy for continuous profiling

Q: My improvements look random?
A: Likely hitting measurement noise from:
   - OS context switching
   - Garbage collection pauses
   - CPU cache effects
   This is why we report statistical measures, not single runs!
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nFull evaluation workflow:")
    print("1. python profile_bottlenecks.py")
    print("2. python main.py --benchmark --iterations 25")
    print("3. python compare_optimization.py") 
    print("4. git checkout develop && python main.py --benchmark --iterations 25")
    print("5. Compare benchmark_results.json from both branches\n")
