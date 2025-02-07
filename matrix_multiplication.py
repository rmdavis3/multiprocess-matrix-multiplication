"""
This project demonstrates the implementation of matrix multiplication using 
Python's multiprocessing capabilities. 

Matrix multiplication is a core operation in many fields such as computer 
science, data analysis, graphics, and machine learning. As the size of 
matrices increases, the computational cost grows exponentially. To address 
this, the project utilizes parallel processing techniques to distribute the 
workload across multiple CPU cores, aiming to reduce execution time and 
improve overall performance.

The program implements different methods for matrix multiplication, including 
a basic approach and two parallelized approaches using Python's 
multiprocessing module: one using apply_async and another using map. This 
allows for the efficient computation of large matrices by breaking down the 
task into smaller, independent operations.
"""

import multiprocessing
import random
import json

from benchmark_config import benchmark_multiplication, BenchmarkConfig


# ---- Helper Functions ----


def can_multiply_matrices(left_matrix, right_matrix):
    """
    Verifies if two matrices can be multiplied based on their dimensions.

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.

    Returns:
    bool: True if multiplication is possible, otherwise False.
    """
    return len(left_matrix[0]) == len(right_matrix)


def initialize_result(left_matrix, right_matrix):
    """
    Initializes the result matrix with zeros based on the dimensions of 
    left_matrix and right_matrix.

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.

    Returns:
    list of list of int: A matrix of zeros with the appropriate size.
    """
    return [[0] * len(right_matrix[0]) for _ in range(len(left_matrix))]


def _calculate_single_row(left_matrix, right_matrix, row_index, result):
    """
    Calculates a single row of the result matrix using the given row from 
    left_matrix and right_matrix.

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.
    row_index (int): The index of the row in left_matrix to compute.
    result (list of list of int): The matrix to store the result.
    """
    for j in range(len(right_matrix[0])):
        total = 0
        for k in range(len(left_matrix[0])):
            total += left_matrix[row_index][k] * right_matrix[k][j]
        result[row_index][j] = total


def _calculate_single_row_for_pool(args):
    """
    Calculates a single row for the result matrix using row-based 
    parallelism.

    Parameters:
    args (tuple): Contains left_matrix, right_matrix, the row index, 
    and the result matrix.
    """
    left_matrix, right_matrix, row_index, result = args
    for j in range(len(right_matrix[0])):
        total = 0
        for k in range(len(left_matrix[0])):
            total += left_matrix[row_index][k] * right_matrix[k][j]
        result[row_index][j] = total


# ---- Matrix Multiplication Functions ----


def multiply_matrices(left_matrix, right_matrix):
    """
    Multiplies two matrices using a basic algorithm.

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.

    Returns:
    list of list of int: The resulting matrix after multiplication.
    """
    if not can_multiply_matrices(left_matrix, right_matrix):
        raise ValueError(
            "Matrix multiplication is not possible: incompatible dimensions.")

    # Initialize the result matrix with zeros
    result = initialize_result(left_matrix, right_matrix)

    # Perform matrix multiplication
    for i, row in enumerate(left_matrix):
        for j in range(len(right_matrix[0])):
            total = 0
            for k in range(len(left_matrix[0])):
                total += left_matrix[i][k] * right_matrix[k][j]
            result[i][j] = total

    return result


def multiply_matrices_using_row_processes_async(left_matrix, right_matrix):
    """
    Multiplies two matrices using multiprocessing with async (each process 
    computes a row).

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.

    Returns:
    list of list of int: The resulting matrix after multiplication.
    """
    if not can_multiply_matrices(left_matrix, right_matrix):
        raise ValueError(
            "Matrix multiplication is not possible: incompatible dimensions.")

    # Initialize the result matrix with zeros
    result = initialize_result(left_matrix, right_matrix)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Start each process to compute a row
    for i in range(len(left_matrix)):
        pool.apply_async(_calculate_single_row, args=(
            left_matrix, right_matrix, i, result))

    pool.close()
    pool.join()

    return result


def multiply_matrices_using_row_processes_map(left_matrix, right_matrix):
    """
    Multiplies two matrices using multiprocessing with map (each process 
    computes a row).

    Parameters:
    left_matrix (list of list of int): The first matrix.
    right_matrix (list of list of int): The second matrix.

    Returns:
    list of list of int: The resulting matrix after multiplication.
    """
    if not can_multiply_matrices(left_matrix, right_matrix):
        raise ValueError(
            "Matrix multiplication is not possible: incompatible dimensions.")

    # Initialize the result matrix with zeros
    result = initialize_result(left_matrix, right_matrix)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Use map to parallelize the computation of rows
    pool.map(_calculate_single_row_for_pool, [
             (left_matrix, right_matrix, i, result) for i in range(len(left_matrix))])

    pool.close()
    pool.join()

    return result


# ---- Matrix Generation and Loading Functions ----


def generate_random_matrix(rows, columns, max_value):
    """
    Generates a random matrix of the given size with values between 1 and 
    max_value.

    Parameters:
    rows (int): The number of rows in the matrix.
    columns (int): The number of columns in the matrix.
    max_value (int): The maximum value for each element in the matrix.

    Returns:
    list of list of int: The generated matrix.
    """
    return [[random.randint(1, max_value) for _ in range(columns)] for _ in range(rows)]


def write_matrix_to_file(matrix, filename):
    """
    Writes the matrix to a JSON file.

    Parameters:
    matrix (list of list of int): The matrix to be written to the file.
    filename (str): The name of the file to save the matrix.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(matrix, file, ensure_ascii=False, indent=4)
        print(f"Matrix has been successfully written to {filename}")
    except IOError as e:
        print(f"Error writing matrix to file: {e}")


def load_matrix(filename):
    """
    Loads a matrix from a JSON file.

    Parameters:
    filename (str): The path to the JSON file containing the matrix.

    Returns:
    list of list of int: The matrix loaded from the file, or None if the file 
    doesn't exist.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The matrix file '{filename}' was not found.")

        return None


# ---- Main Function ----

def main():
    """
    Main program flow
    """

    left_matrix = load_matrix("data/matrix_500x500.json")
    right_matrix = load_matrix("data/matrix_500x500.json")

    if left_matrix and right_matrix:
        print("\nRunning Benchmark...")

        # Create a config for the benchmark
        config = BenchmarkConfig(runs=3)

        # Run and store baseline execution time
        baseline_time = benchmark_multiplication(
            multiply_matrices, left_matrix, right_matrix, "Standard Multiplication", config)

        # Update the config to pass the baseline time
        config.baseline_time = baseline_time
        async_time = benchmark_multiplication(
            multiply_matrices_using_row_processes_async, left_matrix, right_matrix, "Multiprocessing Async", config)

        # Update the config to pass both baseline and async time for comparison
        config.prev_time = async_time
        benchmark_multiplication(multiply_matrices_using_row_processes_map,
                                 left_matrix, right_matrix, "Multiprocessing Map", config)


if __name__ == "__main__":
    main()
