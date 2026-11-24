import numpy as np
import matplotlib.pyplot as plt

def validate(P):
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Matrix is not square")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("Invalid probability")

    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Row sum is not 1")

    return P

def transition(P,n):
    P = validate(P)
    return np.linalg.matrix_power(P,n)

P = np.array([
    [0.7,0.2,0.1],
    [0.1,0.6,0.3],
    [0.4,0.2,0.4]
])

for n in [1,2,5,10,20]:
    print(n,"step transition matrix:")
    print(transition(P,n))

P5 = transition(P,5)
print("Sunny to Rainy in 5 steps:", P5[0,2])

# Plotting
evolution = []

for n in range(1,21):
    evolution.append(transition(P,n)[0])

plt.plot(range(1,21), evolution, marker="o")
plt.title("n-step Transition Probability")
plt.xlabel("Number of steps")
plt.ylabel("Probability")
plt.legend(["Sunny","Cloudy","Rainy"])
plt.show()

print("Probabilities converge to steady state")

