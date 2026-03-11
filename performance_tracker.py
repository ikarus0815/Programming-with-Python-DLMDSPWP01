"""
Performance Tracker Module

A portable performance tracking module for measuring execution time, memory usage,
and operation counts across different phases of the data analysis pipeline.

This module is designed to be easily transferable between different branches
for comparative performance analysis in academic research.

Features:
- Wall-time duration tracking per phase
- Memory usage monitoring (if psutil available)
- Operation counting (e.g., loop iterations, function calls)
- Summary reporting with statistics
- JSON export for further analysis

Usage:
    tracker = PerformanceTracker()
    tracker.start("phase_name")
    # ... code to measure ...
    tracker.end("phase_name")
    tracker.print_summary()
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory tracking disabled.")


@dataclass
class PhaseMetrics:
    """Metrics collected for a single execution phase."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    start_memory: int = 0
    end_memory: int = 0
    peak_memory: int = 0
    memory_delta: int = 0
    operation_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks performance metrics across multiple execution phases.

    Provides timing, memory usage, and operation counting capabilities
    with summary reporting for performance analysis.
    """

    def __init__(self, track_memory: bool = True):
        """
        Initialize the performance tracker.

        Args:
            track_memory: Whether to track memory usage (requires psutil)
        """
        self.phases: Dict[str, PhaseMetrics] = {}
        self.track_memory = track_memory and HAS_PSUTIL
        self.current_phase: Optional[str] = None
        self.execution_order: List[str] = []

        if self.track_memory:
            self.process = psutil.Process()
        else:
            self.process = None

    def start(self, phase_name: str) -> None:
        """
        Start tracking a new phase.

        Args:
            phase_name: Unique identifier for the phase
        """
        if phase_name in self.phases:
            print(f"Warning: Phase '{phase_name}' already exists. Overwriting.")

        metrics = PhaseMetrics()
        metrics.start_time = time.perf_counter()

        if self.track_memory and self.process:
            metrics.start_memory = self.process.memory_info().rss

        self.phases[phase_name] = metrics
        self.current_phase = phase_name

        if phase_name not in self.execution_order:
            self.execution_order.append(phase_name)

    def end(self, phase_name: str) -> None:
        """
        End tracking for a phase.

        Args:
            phase_name: Name of the phase to end
        """
        if phase_name not in self.phases:
            print(f"Warning: Phase '{phase_name}' was not started.")
            return

        metrics = self.phases[phase_name]
        metrics.end_time = time.perf_counter()
        metrics.duration = metrics.end_time - metrics.start_time

        if self.track_memory and self.process:
            metrics.end_memory = self.process.memory_info().rss
            metrics.memory_delta = metrics.end_memory - metrics.start_memory
            # Note: Peak memory tracking would require more frequent sampling

        self.current_phase = None

    def increment_counter(self, phase_name: str, counter_name: str = "operations", increment: int = 1) -> None:
        """
        Increment a custom counter for a phase.

        Args:
            phase_name: Name of the phase
            counter_name: Name of the counter to increment
            increment: Amount to increment by
        """
        if phase_name in self.phases:
            if counter_name not in self.phases[phase_name].custom_metrics:
                self.phases[phase_name].custom_metrics[counter_name] = 0
            self.phases[phase_name].custom_metrics[counter_name] += increment

    def add_metric(self, phase_name: str, metric_name: str, value: Any) -> None:
        """
        Add a custom metric to a phase.

        Args:
            phase_name: Name of the phase
            metric_name: Name of the metric
            value: Metric value
        """
        if phase_name in self.phases:
            self.phases[phase_name].custom_metrics[metric_name] = value

    def get_duration(self, phase_name: str) -> float:
        """
        Get the duration of a completed phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Duration in seconds, or 0.0 if phase not found or not completed
        """
        if phase_name in self.phases:
            return self.phases[phase_name].duration
        return 0.0

    def get_total_duration(self) -> float:
        """
        Get the total duration of all completed phases.

        Returns:
            Total duration in seconds
        """
        return sum(phase.duration for phase in self.phases.values())

    def print_summary(self, sort_by: str = "duration", reverse: bool = True) -> None:
        """
        Print a formatted summary of all tracked phases.

        Args:
            sort_by: Metric to sort by ('duration', 'memory_delta', 'operations')
            reverse: Whether to sort in descending order
        """
        if not self.phases:
            print("No performance data collected.")
            return

        print("\n" + "="*80)
        print("PERFORMANCE TRACKING SUMMARY")
        print("="*80)

        # Sort phases
        if sort_by == "duration":
            sorted_phases = sorted(self.phases.items(),
                                 key=lambda x: x[1].duration,
                                 reverse=reverse)
        elif sort_by == "memory_delta" and self.track_memory:
            sorted_phases = sorted(self.phases.items(),
                                 key=lambda x: x[1].memory_delta,
                                 reverse=reverse)
        else:
            sorted_phases = list(self.phases.items())

        total_duration = self.get_total_duration()

        print(f"Total execution time: {total_duration:.4f} seconds")
        print(f"Number of phases: {len(self.phases)}")
        print(f"Memory tracking: {'Enabled' if self.track_memory else 'Disabled'}")
        print()

        # Header
        header = "| {:<20} | {:>12} | {:>12} | {:>10} |"
        if self.track_memory:
            header = "| {:<20} | {:>12} | {:>12} | {:>10} | {:>12} |"
        print(header.format("Phase", "Duration (s)", "Percentage", "Operations", "Memory (MB)" if self.track_memory else ""))

        separator = "|-" + "-"*20 + "-|-" + "-"*12 + "-|-" + "-"*12 + "-|-" + "-"*10 + "-|"
        if self.track_memory:
            separator = "|-" + "-"*20 + "-|-" + "-"*12 + "-|-" + "-"*12 + "-|-" + "-"*10 + "-|-" + "-"*12 + "-|"
        print(separator)

        # Phase details
        for phase_name, metrics in sorted_phases:
            percentage = (metrics.duration / total_duration * 100) if total_duration > 0 else 0
            operations = metrics.custom_metrics.get("operations", 0)

            row = "| {:<20} | {:>12.4f} | {:>11.1f}% | {:>10} |"
            if self.track_memory:
                memory_mb = metrics.memory_delta / (1024 * 1024) if metrics.memory_delta else 0
                row = "| {:<20} | {:>12.4f} | {:>11.1f}% | {:>10} | {:>+11.2f} |"

            print(row.format(
                phase_name[:20],
                metrics.duration,
                percentage,
                operations,
                memory_mb if self.track_memory else ""
            ))

        print(separator)
        print()

        # Custom metrics summary
        custom_found = any(metrics.custom_metrics for metrics in self.phases.values())
        if custom_found:
            print("Custom Metrics:")
            for phase_name, metrics in sorted_phases:
                if metrics.custom_metrics:
                    print(f"  {phase_name}:")
                    for key, value in metrics.custom_metrics.items():
                        if key != "operations":  # Already shown in table
                            print(f"    {key}: {value}")
            print()

    def export_json(self, filename: str) -> None:
        """
        Export performance data to a JSON file.

        Args:
            filename: Output filename
        """
        data = {
            "metadata": {
                "total_duration": self.get_total_duration(),
                "phase_count": len(self.phases),
                "memory_tracking": self.track_memory,
                "execution_order": self.execution_order
            },
            "phases": {}
        }

        for phase_name, metrics in self.phases.items():
            data["phases"][phase_name] = {
                "duration": metrics.duration,
                "start_time": metrics.start_time,
                "end_time": metrics.end_time,
                "memory_delta": metrics.memory_delta if self.track_memory else None,
                "custom_metrics": metrics.custom_metrics
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Performance data exported to {filename}")

    def reset(self) -> None:
        """Reset all collected performance data."""
        self.phases.clear()
        self.execution_order.clear()
        self.current_phase = None


# Convenience functions for quick timing
def time_function(func):
    """
    Decorator to time a function execution.

    Usage:
        @time_function
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def benchmark_function(func, iterations: int = 1):
    """
    Benchmark a function over multiple iterations.

    Args:
        func: Function to benchmark
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "iterations": iterations,
        "total_time": sum(times),
        "average_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "times": times
    }#</content>
#<parameter name="filePath">c:\Users\marti\python\Github\Programming_with_Python_DLMDSPWP01\performance_tracker.py