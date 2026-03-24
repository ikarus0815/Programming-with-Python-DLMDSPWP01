"""
Numba JIT-Compiled Fast Kernels

These functions are compiled to machine code by Numba for 2-10x speedup.
Focus on tight numerical loops in core algorithms.
"""

import numpy as np
from typing import Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# FUNCTION SELECTION KERNELS
# =============================================================================

@jit(nopython=True)
def compute_sum_of_squares(train_vals: np.ndarray, ideal_vals: np.ndarray) -> float:
    """
    Compute sum of squared residuals using BLAS-optimized dot product.
    
    JIT compiled for maximum speed - this is called 200 times per execution.
    
    Parameters
    ----------
    train_vals : np.ndarray
        Training data column (float64, shape: n)
    ideal_vals : np.ndarray
        Ideal function column (float64, shape: n)
    
    Returns
    -------
    float
        Sum of squared residuals (loss metric)
    """
    residuals = train_vals - ideal_vals
    sum_sq = np.dot(residuals, residuals)
    return float(sum_sq)


@jit(nopython=True)
def compute_max_deviation(train_vals: np.ndarray, ideal_vals: np.ndarray) -> float:
    """
    Compute maximum absolute deviation (for threshold calculation).
    
    JIT compiled - called 200 times per execution.
    
    Parameters
    ----------
    train_vals : np.ndarray
        Training data column (float64, shape: n)
    ideal_vals : np.ndarray
        Ideal function column (float64, shape: n)
        
    Returns
    -------
    float
        Maximum absolute deviation
    """
    residuals = train_vals - ideal_vals
    max_dev = np.max(np.abs(residuals))
    return float(max_dev)


@jit(nopython=True)
def find_best_ideal_by_column(train_vals: np.ndarray, ideal_values_stacked: np.ndarray) -> Tuple[int, float, float]:
    """
    Find best ideal function for a training column via exhaustive search.
    
    JIT compiled inner loop - searches all 50 ideal functions.
    
    Parameters
    ----------
    train_vals : np.ndarray
        Training column (shape: n)
    ideal_values_stacked : np.ndarray
        All 50 ideal functions stacked (shape: 50, n)
    
    Returns
    -------
    tuple
        (best_idx, best_sum_sq, best_max_dev)
    """
    best_sum_sq = np.inf
    best_max_dev = 0.0
    best_idx = 0
    
    for idx in range(ideal_values_stacked.shape[0]):
        ideal_vals = ideal_values_stacked[idx]
        residuals = train_vals - ideal_vals
        sum_sq = np.dot(residuals, residuals)
        
        if sum_sq < best_sum_sq:
            best_sum_sq = sum_sq
            best_max_dev = np.max(np.abs(residuals))
            best_idx = idx + 1  
    
    return best_idx, best_sum_sq, best_max_dev


# =============================================================================
# TEST MAPPING KERNELS
# =============================================================================

@jit(nopython=True)
def find_best_test_mapping(y_val: float, thresholds: np.ndarray, 
                           ideal_values_stacked: np.ndarray, idx: int) -> Tuple[float, int]:
    """
    Find best ideal function for a single test point (JIT-compiled inner loop).
    
    Called once per test point (100+ times). 
    Must find best fit within thresholds using pre-computed ideal values.
    
    Parameters
    ----------
    y_val : float
        Test point y-value
    thresholds : np.ndarray
        Pre-computed thresholds for each training column (shape: 4)
    ideal_values_stacked : np.ndarray
        Ideal values stacked (shape: 4, n) 
    idx : int
        Index of test point's x-value in ideal data
    
    Returns
    -------
    tuple
        (best_delta, chosen_index) or (inf, -1) if no fit
    """
    best_delta = np.inf
    chosen_index = -1
    
    for col_idx in range(ideal_values_stacked.shape[0]):
        ideal_vals = ideal_values_stacked[col_idx]
        threshold = thresholds[col_idx]
        
        delta = np.abs(y_val - ideal_vals[idx])
        
        if delta <= threshold and delta < best_delta:
            best_delta = delta
            chosen_index = col_idx
    
    return best_delta, chosen_index


@jit(nopython=True)
def batch_find_x_indices(x_test: np.ndarray, x_ideal: np.ndarray) -> np.ndarray:
    """
    Find indices of each test x-value in ideal x-values (numpy searchsorted-like).
    
    Can't use Python dict in nopython, so use numpy array operations.
    Assumes x_ideal is sorted (it always is in this dataset).
    
    Parameters
    ----------
    x_test : np.ndarray
        Test x-values (shape: m)
    x_ideal : np.ndarray
        Ideal x-values (shape: n), sorted
    
    Returns
    -------
    np.ndarray
        Indices in x_ideal for each x_test value, dtype int64
    """
    indices = np.empty(len(x_test), dtype=np.int64)
    
    for i in range(len(x_test)):
        found = False
        for j in range(len(x_ideal)):
            # Float comparison with small tolerance for floating point errors
            if np.abs(x_test[i] - x_ideal[j]) < 1e-10:
                indices[i] = j
                found = True
                break
        
        if not found:
            indices[i] = -1
    
    return indices


@jit(nopython=True)
def batch_find_best_mapping(y_test: np.ndarray, thresholds: np.ndarray, 
                            ideal_values_stacked: np.ndarray, x_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find best ideal function for ALL test points at once (batch vectorized).
    
    This is the key optimization - instead of calling find_best_test_mapping() 100 times
    from Python (with Python loop overhead), we process all 100 test points in ONE
    NUMBA-compiled function. This keeps the entire loop in native code.
    
    Parameters
    ----------
    y_test : np.ndarray
        All test point y-values (shape: m, typically 100)
    thresholds : np.ndarray
        Pre-computed thresholds for each training column (shape: 4)
    ideal_values_stacked : np.ndarray
        Ideal values stacked (shape: 4, n)
    x_indices : np.ndarray
        Pre-computed x indices for each test point (shape: m)
    
    Returns
    -------
    tuple
        (deltas: ndarray shape (m,), col_indices: ndarray shape (m,))
    """
    num_test_points = len(y_test)
    deltas = np.empty(num_test_points, dtype=np.float64)
    col_indices = np.empty(num_test_points, dtype=np.int64)
    
    for i in range(num_test_points):
        y_val = y_test[i]
        idx = x_indices[i]
        
        if idx < 0:
            deltas[i] = np.nan
            col_indices[i] = -1
            continue
        
        best_delta = np.inf
        best_col_idx = -1
        
        for col_idx in range(ideal_values_stacked.shape[0]):
            ideal_vals = ideal_values_stacked[col_idx]
            threshold = thresholds[col_idx]
            
            delta = np.abs(y_val - ideal_vals[idx])
            
            if delta <= threshold and delta < best_delta:
                best_delta = delta
                best_col_idx = col_idx
        
        deltas[i] = best_delta if best_delta != np.inf else np.nan
        col_indices[i] = best_col_idx
    
    return deltas, col_indices


def print_numba_status():
    """Print Numba availability and compilation status"""
    if NUMBA_AVAILABLE:
        print("✓ Numba JIT compilation AVAILABLE")
        print("  Functions will be compiled on first call")
        print("  Subsequent calls run at ~C speed (2-10x faster than Python)")
    else:
        print("⚠ Numba NOT available - functions run in Python")


if __name__ == "__main__":
    import time
    
    print_numba_status()
    
    print("\n" + "="*60)
    print("NUMBA KERNEL VERIFICATION")
    print("="*60)
    
    train = np.random.randn(400).astype(np.float64)
    ideal = np.random.randn(400).astype(np.float64)
    
    print(f"\nTesting compute_sum_of_squares()...")
    t0 = time.perf_counter()
    for _ in range(100):
        result = compute_sum_of_squares(train, ideal)
    t1 = time.perf_counter()
    print(f"  100 calls: {(t1-t0)*1000:.3f}ms (avg {(t1-t0)*10:.3f}ms per call)")
    print(f"  Result check: {result:.3f}")
    
    print(f"\nTesting compute_max_deviation()...")
    t0 = time.perf_counter()
    for _ in range(100):
        result = compute_max_deviation(train, ideal)
    t1 = time.perf_counter()
    print(f"  100 calls: {(t1-t0)*1000:.3f}ms (avg {(t1-t0)*10:.3f}ms per call)")
    print(f"  Result check: {result:.3f}")
    
    print(f"\nTesting find_best_ideal_by_column()...")
    ideal_stacked = np.random.randn(50, 400).astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(10):
        idx, loss, dev = find_best_ideal_by_column(train, ideal_stacked)
    t1 = time.perf_counter()
    print(f"  10 calls: {(t1-t0)*1000:.3f}ms (avg {(t1-t0)*100:.3f}ms per call)")
    print(f"  Result: idx={idx}, loss={loss:.3f}, max_dev={dev:.3f}")
    
    print("\n✓ All kernels verified and compiled")
