import numpy as np
from scipy.stats import f

A = [10.2, 9.8, 10.5, 10.1, 9.9, 10.3, 10.0]
B = [9.5, 10.6, 9.2, 10.8, 9.4, 10.9, 9.1]

v1 = np.var(A, ddof=1)
v2 = np.var(B, ddof=1)

if v1 >= v2:
    F = v1/v2
    df1 = len(A)-1
    df2 = len(B)-1
else:
    F = v2/v1
    df1 = len(B)-1
    df2 = len(A)-1

alpha = 0.05

a = f.cdf(F, df1, df2)
p = 2*min(a, 1-a)

critical = f.ppf(1-alpha/2, df1, df2)

print("Variance 1:", v1)
print("Variance 2:", v2)
print("F statistic:", F)
print("Degrees of freedom:", df1, df2)
print("Critical value:", critical)
print("P value:", p)

if p <= alpha:
    print("Reject H0")
    print("The machines differ significantly in consistency")
else:
    print("Fail to reject H0")
    print("No significant difference in consistency")

