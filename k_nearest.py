#Question 4: Develop a program to implement K-Nearest Neighbor algorithm to classify the iris data set. Print both correct and wrong predictions.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import datasets

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class of each sample in X"""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Predict the class of a single sample"""
        # Calculate distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]

        # Get indices of k nearest examples
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest examples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
knn = KNN(k=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Print correct and wrong predictions
print("\nPredictions:")
print("Sample\tActual\t\tPredicted\tCorrect")
print("-"*50)

for i in range(len(y_test)):
    actual_class = target_names[y_test[i]]
    predicted_class = target_names[y_pred[i]]
    is_correct = y_test[i] == y_pred[i]

    print(f"{i}\t{actual_class}\t{predicted_class}\t{'✓' if is_correct else '✗'}")

# Count correct and wrong predictions
correct_count = sum(y_test == y_pred)
wrong_count = sum(y_test != y_pred)

print(f"\nCorrect predictions: {correct_count} ({correct_count/len(y_test)*100:.2f}%)")
print(f"Wrong predictions: {wrong_count} ({wrong_count/len(y_test)*100:.2f}%)")

# Visualize the predictions (first two features)
plt.figure(figsize=(10, 6))

# Plot correct predictions
mask_correct = y_test == y_pred
plt.scatter(X_test[mask_correct, 0], X_test[mask_correct, 1],
           c=y_test[mask_correct], marker='o', edgecolor='k',
           alpha=0.7, label='Correct')

# Plot wrong predictions
mask_wrong = y_test != y_pred
if any(mask_wrong):  # Check if there are any wrong predictions
    plt.scatter(X_test[mask_wrong, 0], X_test[mask_wrong, 1],
               c=y_test[mask_wrong], marker='X', edgecolor='red', s=100,
               alpha=0.9, label='Wrong')

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('KNN Classification Results (k=3)')
plt.legend()
plt.colorbar(label='Class')
plt.show()
