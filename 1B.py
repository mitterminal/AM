# -----------------------------------------------------------
# 1B: Eigenvalues / Eigenvectors and Diagonalization
# -----------------------------------------------------------
import numpy as np

A = np.array([
    [4, 1, 0],
    [2, 3, 0],
    [0, 0, 5]
], dtype=float)

# Step 1: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Matrix A:\n", A)
print("\nEigenvalues (lambda):", eigenvalues)
print("\nEigenvectors (as columns of P):\n", eigenvectors)

# Step 2: Build P (eigenvector matrix) and D (diagonal eigenvalue matrix)
P = eigenvectors
D = np.diag(eigenvalues)

# Step 3: Diagonalize: A = P D P^-1  --> verify by reconstructing A
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print("\nDiagonal matrix D:\n", D)
print("Reconstructed A:\n", A_reconstructed)
print("\nMatches original A? ", np.allclose(A, A_reconstructed))

# Step 4: Practical payoff of diagonalization - fast matrix powers, e.g. A^5
A_power5_direct = np.linalg.matrix_power(A, 5)
A_power5_via_D  = P @ D**5 @ P_inv

print("\nA^5 computed directly:\n", A_power5_direct)
print("\nA^5 computed via diagonalization (P D^5 P^-1):\n", A_power5_via_D.real)

