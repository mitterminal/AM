import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# Binomial distribution
n, p = 10, 0.4
k = np.arange(n+1)
pmf = binom.pmf(k, n, p)

for i in range(n+1):
    print("P(X =", i, ") =", round(pmf[i], 4))

s = np.random.binomial(n, p, 10000)

print("Theoretical mean:", n*p)
print("Simulated mean:", round(np.mean(s), 4))
print("Theoretical variance:", n*p*(1-p))
print("Simulated variance:", round(np.var(s), 4))

# Normal distribution
x = np.linspace(-4, 4, 400)
pdf = norm.pdf(x)

print("P(-1 < X < 1):", round(norm.cdf(1)-norm.cdf(-1), 4))

# Plotting
plt.figure(figsize=(12, 4))

# Binomial distribution
plt.subplot(1, 2, 1)
plt.bar(k, pmf)
plt.title("Binomial Distribution")
plt.xlabel("Number of successes (k)")
plt.ylabel("P(X = k)")

# Normal distribution
plt.subplot(1, 2, 2)
plt.plot(x, pdf, color="orange")
plt.fill_between(x, pdf, where=(x>=-1)&(x<=1),
                 color="orange", alpha=0.3)
plt.title("Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")

plt.tight_layout()
plt.show()

