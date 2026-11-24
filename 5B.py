import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n = 120

size = np.random.randint(500,4000,n)
bed = np.random.randint(1,6,n)
bath = np.random.randint(1,4,n)
dist = np.random.randint(1,30,n)

price = 200*size + 3000*bed + 5000*bath - 1500*dist
price = price + np.random.randint(-10000,10000,n)

data = pd.DataFrame({
    "Size": size,
    "Bedrooms": bed,
    "Bathrooms": bath,
    "Distance": dist,
    "Price": price
})

print(data.head())

# Features and target
X = data[["Size","Bedrooms","Bathrooms","Distance"]]
y = data["Price"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test,y_pred))
print("R2 Score:", r2_score(y_test,y_pred))

# Regression equation
a,b,c,d = model.coef_

print("Price =", model.intercept_, "+",
      a,"* Size +", b,"* Bedrooms +",
      c,"* Bathrooms +", d,"* Distance")

# Actual vs Predicted graph
plt.scatter(y_test,y_pred)
plt.plot([min(y),max(y)], [min(y),max(y)], "r--")
plt.title("Actual vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# Residual graph
residual = y_test-y_pred

plt.scatter(y_pred,residual)
plt.axhline(0,color="red")
plt.title("Residual Plot")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.show()

