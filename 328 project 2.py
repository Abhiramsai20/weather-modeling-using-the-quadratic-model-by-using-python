import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# === 1. Sample Data: Hour of day (0–23) and corresponding temperatures (in °C) ===
t = np.array([0, 3, 6, 9, 12, 15, 18, 21, 23])        # Time of day
T = np.array([18, 17, 16, 20, 25, 27, 24, 21, 19])     # Temperature

# === 2. Create Quadratic Features ===
t2 = t ** 2
X = np.column_stack((t2, t, np.ones_like(t)))  # Features: [t^2, t, 1]

# === 3. Fit the Quadratic Model ===
model = LinearRegression()
model.fit(X, T)

# Extract coefficients
a, b = model.coef_[0], model.coef_[1]
c = model.intercept_
print(f"Quadratic Model: T(t) = {a:.4f}·t² + {b:.4f}·t + {c:.4f}")

# === 4. Predict for a Smooth Curve ===
t_pred = np.linspace(0, 23, 100)
X_pred = np.column_stack((t_pred**2, t_pred, np.ones_like(t_pred)))
T_pred = model.predict(X_pred)

# === 5. Plot Observed vs. Predicted ===
plt.figure(figsize=(10, 5))
plt.scatter(t, T, color='red', label='Observed Data')
plt.plot(t_pred, T_pred, color='blue', label='Quadratic Fit')
plt.title('Temperature Prediction Using Quadratic Model')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# === 6. Evaluate Model Accuracy ===
T_fit = model.predict(X)
mse = mean_squared_error(T, T_fit)
print(f"Mean Squared Error: {mse:.2f}")
