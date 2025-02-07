"""
This module contains functionality for benchmarking matrix multiplication 
functions.
It measures execution time and compares the performance of different methods
in terms of speedup and efficiency. The benchmark is performed over multiple 
runs to account for variability in execution times.
"""

import time
try:
    import psutil
except ImportError:
    psutil = None
    print("psutil module is not available, CPU usage tracking will be skipped.")


class BenchmarkConfig:
    """
    A configuration class to store parameters for benchmarking matrix 
    multiplication functions.
    """

    def __init__(self, runs=3, baseline_time=None, prev_time=None):
        """
        Initializes the BenchmarkConfig instance with the given parameters.

        Parameters:
        - runs (int): The number of runs to perform. Default is 3.
        - baseline_time (float or None): The baseline execution time to 
        compare against. Default is None.
        - prev_time (float or None): The previous method's execution time to 
        compare against. Default is None.
        """
        self.runs = runs
        self.baseline_time = baseline_time
        self.prev_time = prev_time


def benchmark_multiplication(func, left_matrix, right_matrix, method_name, config):
    """
    Benchmarks a matrix multiplication function by running it multiple times 
    and reporting performance metrics.

    This function calculates the average execution time over multiple runs and 
    compares the performance of the method against a baseline and/or previous 
    method's execution time if provided.

    Parameters:
    - func (function): The matrix multiplication function to benchmark.
    - left_matrix (list of list of int): The first matrix to multiply.
    - right_matrix (list of list of int): The second matrix to multiply.
    - method_name (str): A name or label for the method being benchmarked.
    - config (BenchmarkConfig): The configuration object containing benchmarking 
      parameters such as the number of runs, baseline time, and previous method 
      execution time.

    Returns:
    - float: The average execution time of the benchmarked method.
    """

    execution_times = []
    cpu_usages = []  # List to store CPU usage for each run

    for _ in range(config.runs):
        start_time = time.perf_counter()

        # Start tracking CPU usage
        process = psutil.Process()
        cpu_start = process.cpu_percent(interval=None)

        func(left_matrix, right_matrix)  # Run the function

        end_time = time.perf_counter()
        cpu_end = process.cpu_percent(interval=None)

        execution_times.append(end_time - start_time)
        # Calculate CPU usage during the run
        cpu_usages.append(cpu_end - cpu_start)

    # Compute average execution time and CPU usage
    avg_time = sum(execution_times) / config.runs
    avg_cpu_usage = sum(cpu_usages) / config.runs

    # Calculate Speedup Factor if baseline is provided
    speedup_baseline = config.baseline_time / \
        avg_time if config.baseline_time else None

    # Calculate Speedup Factor if previous time is provided
    speedup_prev = config.prev_time / avg_time if config.prev_time else None

    print(f"\n{method_name} Performance Metrics (Runs: {config.runs}):")
    print(f"- Average Execution Time: {avg_time:.6f} seconds")
    print(f"- Average CPU Usage: {avg_cpu_usage:.2f}%")
    if speedup_baseline:
        print(f"- Speedup Factor (vs Baseline): {speedup_baseline:.2f}x")
    if speedup_prev:
        print(f"- Speedup Factor (vs Async): {speedup_prev:.2f}x")

    return avg_time  # Return avg_time to use as a baseline for comparisons
