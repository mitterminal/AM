import numpy as np
from scipy.stats import norm

np.random.seed(1)
data = np.random.normal(50.6, 2, 40)

mu = 50
n = len(data)
xbar = np.mean(data)
sigma = np.std(data, ddof=1)

alpha = 0.05

z = (xbar-mu)/(sigma/np.sqrt(n))
p = 2*(1-norm.cdf(abs(z)))

print("Sample size:", n)
print("Sample mean:", xbar)
print("Hypothesized mean:", mu)
print("Z statistic:", z)
print("P value:", p)
print("Alpha:", alpha)

print("\nInterpretation:")

if p <= alpha:
    print("Reject H0")
    print("We reject the claim that the mean is 50 ml")
else:
    print("Fail to reject H0")
    print("We do not reject the claim that the mean is 50 ml")

