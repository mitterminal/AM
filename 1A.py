# -----------------------------------------------------------
# 1A: Matrix Addition, Multiplication, and Scalar Operations
# -----------------------------------------------------------

# ---- Pure-Python implementation (shows the underlying mechanics) ----

def add_matrices(A, B):
    """Element-wise addition of two matrices of the same shape."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def multiply_matrices(A, B):
    """Standard row-by-column matrix multiplication A (m x n) . B (n x p)."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Number of columns of A must equal number of rows of B"
    result = [[0] * cols_B for i in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result

def scalar_multiply(matrix, scalar):
    """Multiply every entry of the matrix by a scalar."""
    return [[scalar * x for x in row] for row in matrix]

# Example matrices
A = [[2, 3], [4, 5]]
B = [[1, 0], [2, 1]]
scalar = 3

print("Matrix A         :", A)
print("Matrix B         :", B)
print("A + B            :", add_matrices(A, B))
print("A x B (A*B)      :", multiply_matrices(A, B))
print("B x A (B*A)      :", multiply_matrices(B, A), "<-- shows AB != BA")
print(f"{scalar} x A            :", scalar_multiply(A, scalar))

# ---- Verification using NumPy (the standard tool used in practice) ----
import numpy as np
An = np.array(A)
Bn = np.array(B)
print("\n--- NumPy verification ---")
print("A + B (numpy):\n", An + Bn)
print("A @ B (numpy):\n", An @ Bn)   # matrix multiplication
print("A * B (numpy, element-wise!):\n", An * Bn)  # NOTE: '*' is elementwise, NOT matrix mult
print(f"{scalar} * A (numpy):\n", scalar * An)

