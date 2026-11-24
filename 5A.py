import numpy as np
import matplotlib.pyplot as plt

x = np.array([2,4,6,8,10,3,7,5,9,1])
y = np.array([50,60,68,82,90,55,75,65,88,40])

# Pearson correlation
a = x - np.mean(x)
b = y - np.mean(y)

num = np.sum(a*b)
den = np.sqrt(np.sum(a**2)*np.sum(b**2))

r = num/den

print("Manual correlation:", r)
print("NumPy correlation:", np.corrcoef(x,y)[0,1])

# Interpretation
if r > 0.7:
    print("Strong positive correlation")
elif r > 0.3:
    print("Moderate positive correlation")
elif r > -0.3:
    print("Weak or no correlation")
elif r > -0.7:
    print("Moderate negative correlation")
else:
    print("Strong negative correlation")

# Plotting
plt.scatter(x,y,label="Data points")

m,c = np.polyfit(x,y,1)
plt.plot(x,m*x+c,"r--",label="Best fit-line")

plt.title("Karl Pearson Correlation")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.legend()
plt.show()

