# Question 5: Develop a program to implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class LocallyWeightedRegression:
    def __init__(self, tau=1.0):
        """
        Initialize Locally Weighted Regression model

        Args:
            tau: Bandwidth parameter that controls the width of the kernel (smoothing parameter)
        """
        self.tau = tau
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Store the training data

        Args:
            x: Training features
            y: Target values
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x_query):
        """
        Make predictions for query points using locally weighted regression

        Args:
            x_query: Points to make predictions for

        Returns:
            Predicted values for query points
        """
        # If input is a single point, convert to array
        if not isinstance(x_query, np.ndarray):
            x_query = np.array([x_query])
        elif x_query.ndim == 1:
            x_query = x_query.reshape(-1, 1)

        # Make predictions for each query point
        predictions = np.zeros(len(x_query))

        for i, x_q in enumerate(x_query):
            # Calculate weights for all training points based on distance to query point
            weights = self._calculate_weights(x_q)

            # Create weight matrix W
            W = np.diag(weights)

            # Add constant term (intercept) to x_train
            X = np.column_stack([np.ones(len(self.x_train)), self.x_train])

            # Calculate weighted least squares solution: β = (X^T W X)^(-1) X^T W y
            try:
                xTWx = X.T @ W @ X
                xTWy = X.T @ W @ self.y_train
                beta = np.linalg.inv(xTWx) @ xTWy

                # Make prediction for query point: y = β₀ + β₁x
                predictions[i] = beta[0] + beta[1] * x_q
            except np.linalg.LinAlgError:
                # Handle singular matrix error
                predictions[i] = np.mean(self.y_train)

        return predictions

    def _calculate_weights(self, x_query):
        """
        Calculate weights for all training points based on distance to query point
        using Gaussian kernel

        Args:
            x_query: Query point

        Returns:
            Array of weights for each training point
        """
        # Compute squared distances
        distances = np.sum((self.x_train - x_query) ** 2, axis=1)

        # Apply Gaussian kernel
        # w(x) = exp(-||x - x_query||^2 / (2*τ^2))
        weights = np.exp(-distances / (2 * self.tau**2))

        return weights


# Generate synthetic dataset
def generate_1d_data(n=100, noise=0.5):
    x = np.linspace(0, 10, n)
    y = np.sin(x) + np.random.normal(0, noise, n)
    return x.reshape(-1, 1), y

def generate_2d_data(n=100, noise=0.5):
    x1 = np.random.uniform(0, 5, n)
    x2 = np.random.uniform(0, 5, n)
    y = np.sin(x1) * np.cos(x2) + np.random.normal(0, noise, n)
    return np.column_stack([x1, x2]), y


# 1D Example
x_1d, y_1d = generate_1d_data(200, noise=0.3)

# Initialize models with different bandwidths
models_1d = [
    LocallyWeightedRegression(tau=0.1),  # Very local (underfitting)
    LocallyWeightedRegression(tau=0.5),  # Good balance
    LocallyWeightedRegression(tau=2.0)   # Too global (overfitting)
]

# Fit models
for model in models_1d:
    model.fit(x_1d, y_1d)

# Create test points
x_test_1d = np.linspace(0, 10, 1000).reshape(-1, 1)

# Visualize results
plt.figure(figsize=(12, 6))
plt.scatter(x_1d, y_1d, color='blue', alpha=0.5, label='Training data')
plt.plot(x_test_1d, np.sin(x_test_1d), 'g-', label='True function')

for i, model in enumerate(models_1d):
    y_pred = model.predict(x_test_1d)
    plt.plot(x_test_1d, y_pred, label=f'LWR (tau={model.tau})')

plt.title('Locally Weighted Regression - 1D Example')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('LWR_1D.png')
plt.show()


# 2D Example - Visualize with 3D surface
x_2d, y_2d = generate_2d_data(500, noise=0.2)

# Fit model
model_2d = LocallyWeightedRegression(tau=0.7)
model_2d.fit(x_2d, y_2d)

# Create a grid of test points
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 5, 50), np.linspace(0, 5, 50))
x_test_2d = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

# Get predictions for test points
y_pred_2d = model_2d.predict(x_test_2d)
y_pred_2d_grid = y_pred_2d.reshape(x1_grid.shape)

# True function
def true_function(x1, x2):
    return np.sin(x1) * np.cos(x2)

y_true_grid = true_function(x1_grid, x2_grid)

# 3D Plot
fig = plt.figure(figsize=(18, 8))

# Original scatter plot
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(x_2d[:, 0], x_2d[:, 1], y_2d, c=y_2d, cmap=cm.viridis, alpha=0.7)
ax1.set_title('Training Data')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

# True function surface
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(x1_grid, x2_grid, y_true_grid, cmap=cm.viridis, alpha=0.7, antialiased=True)
ax2.set_title('True Function: sin(x1) * cos(x2)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

# LWR prediction surface
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(x1_grid, x2_grid, y_pred_2d_grid, cmap=cm.viridis, alpha=0.7, antialiased=True)
ax3.set_title(f'LWR Predictions (tau={model_2d.tau})')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')

plt.tight_layout()
plt.savefig('LWR_2D.png')
plt.show()

# Check prediction error
mse = np.mean((y_pred_2d - true_function(x_test_2d[:, 0], x_test_2d[:, 1]))**2)
print(f"Mean Squared Error on test grid: {mse:.6f}")
