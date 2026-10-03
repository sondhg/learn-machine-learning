import matplotlib.pyplot as plt
import numpy as np

# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)  # each row is one data point

# Calculating weights of the linear regression model
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# weights
w_0, w_1 = w[0], w[1]

y1 = w_1 * 155 + w_0
y2 = w_1 * 160 + w_0

print(f"Input 155cm, true output 52kg, predicted output {y1:.2f}kg.")
print(f"Input 160cm, true output 56kg, predicted output {y2:.2f}kg.")

# Plot training data
plt.scatter(X, y, label="Training data")

# Plot the linear regression model
x_line = np.linspace(start=X.min(), stop=X.max(), num=100)
y_line = w_1 * x_line + w_0

plt.plot(x_line, y_line, label="Linear regression")

plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Predict weight based on height")
plt.grid()
plt.legend()
plt.show()
