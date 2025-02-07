# Multiprocess Matrix Multiplication

This project benchmarks different matrix multiplication techniques in Python, comparing the performance of standard and parallelized methods.

## Overview

Matrix multiplication is a fundamental operation used in various fields such as computer science, data analysis, graphics, and machine learning. As the size of matrices increases, the computational cost grows exponentially. To address this, the project utilizes parallel processing techniques to distribute the workload across multiple CPU cores, aiming to reduce execution time and improve overall performance.

This project benchmarks the performance of three different matrix multiplication methods:

- **Basic Matrix Multiplication**: The standard, non-parallel algorithm for matrix multiplication.
- **Multiprocessing with Async**: Each row of the result matrix is computed in a separate process using asynchronous calls.
- **Multiprocessing with Map**: Uses Python’s `map` function to parallelize the computation, allowing for better performance on larger matrices.

The project also includes benchmarking functionality to compare the performance of each method across multiple runs, CPU usage, and computes speedup factors for the parallelized approaches.

## Features

- **Matrix Generation**: Generate random matrices for testing.
- **Matrix Multiplication Methods**: Basic, async-based, and map-based methods.
- **Benchmarking**: Measures average execution time, CPU usage, and speedup factors.

## Matrix Multiplication Methods
This project includes three different methods for matrix multiplication:

1. Basic Matrix Multiplication<br>
This is the standard, sequential algorithm for matrix multiplication. It processes each element of the result matrix one by one, using a triple nested loop to multiply corresponding elements from the two input matrices.

4. Multiprocessing with Async<br>
This method divides the computation by assigning each row of the result matrix to a separate process. It uses asynchronous calls to perform each row's calculation concurrently, which helps in speeding up the operation, especially for larger matrices.

5. Multiprocessing with Map<br>
This method uses Python’s `map` function to parallelize the matrix multiplication. The computation is split across multiple processes, where each process computes a row of the result matrix. This approach is often more efficient than the async method, as it leverages the parallelism more effectively.
 
## Usage

1. Install dependencies:
    ```bash
    pip install psutil
    ```
2. Generate matrices and benchmark methods:
    ```python
    # Example usage of benchmarking
    """
    Parameters:
    - func (function): The matrix multiplication function to benchmark.
    - left_matrix (list of list of int): The first matrix to multiply.
    - right_matrix (list of list of int): The second matrix to multiply.
    - method_name (str): A name or label for the method being benchmarked.
    - config (BenchmarkConfig): The configuration object containing benchmarking 
      parameters such as the number of runs, baseline time, and previous method 
      execution time.
    """
    benchmark_multiplication(func, left_matrix, right_matrix, method_name, config)
    ```
    
3. Sample output:
    ```bash
    Running Benchmark...
    
    Standard Multiplication Performance Metrics (Runs: 3):
    - Average Execution Time: 6.637986 seconds
    - Average CPU Usage: 99.70%
    
    Multiprocessing Async Performance Metrics (Runs: 3):
    - Average Execution Time: 3.488972 seconds
    - Average CPU Usage: 88.60%
    - Speedup Factor (vs Baseline): 1.90x
    
    Multiprocessing Map Performance Metrics (Runs: 3):
    - Average Execution Time: 1.210284 seconds
    - Average CPU Usage: 28.20%
    - Speedup Factor (vs Baseline): 5.48x
    - Speedup Factor (vs Async): 2.88x
    ```

