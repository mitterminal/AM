import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

population = np.random.uniform(0, 1, 100000)

means = []

for _ in range(1000):
    sample = np.random.choice(population, 30)
    means.append(np.mean(sample))

m = np.mean(means)
s = np.std(means)
se = np.std(population)/np.sqrt(30)

print("Population mean:", np.mean(population))
print("Population std:", np.std(population))
print("Sample means:", len(means))
print("Mean of means:", m)
print("Std of means:", s)
print("Standard error:", se)

# Plotting
x = np.linspace(min(means), max(means), 500)
pdf = norm.pdf(x, m, s)

plt.hist(means, bins=30, density=True,
         edgecolor="black", label="Sample means")

plt.plot(x, pdf, color="red", label="Normal curve")

plt.title("Central Limit Theorem")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.show()

