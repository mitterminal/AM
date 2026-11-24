import numpy as np
import matplotlib.pyplot as plt

def lagrange(x, y, a):
    result = 0

    for i in range(len(x)):
        term = y[i]

        for j in range(len(x)):
            if i != j:
                term = term * (a-x[j])/(x[i]-x[j])

        result += term

    return result

x = np.array([1,2,3,4,5])
y = np.array([2,4,1,3,7])

a = 2.5
b = lagrange(x,y,a)

print("Interpolated value:", b)

# Plotting
X = np.linspace(min(x),max(x),200)
Y = [lagrange(x,y,i) for i in X]

plt.scatter(x,y,color="red")
plt.plot(X,Y)
plt.scatter(a,b,color="green",marker="x")

plt.title("Lagrange Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Known points","Interpolation curve","Interpolated point"])
plt.show()

