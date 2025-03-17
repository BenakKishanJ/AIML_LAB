# Question 6: Develop a program to implement the naive Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier few test data sets.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.mean = {}
        self.variance = {}
        self.n_features = None

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier

        Args:
            X: Training features
            y: Training labels
        """
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n_samples = X.shape[0]

        # Calculate class priors P(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

        # Calculate mean and variance for each feature in each class
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0) + 1e-9  # Add small value to avoid division by zero

    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate Gaussian likelihood P(x|y) for continuous features

        Args:
            x: Feature value
            mean: Mean of the feature for a specific class
            var: Variance of the feature for a specific class

        Returns:
            Probability density
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return exponent / np.sqrt(2 * np.pi * var)

    def _calculate_class_probability(self, x, c):
        """
        Calculate class probability P(y|x) for a single sample

        Args:
            x: Sample features
            c: Class label

        Returns:
            Probability of class c given features x
        """
        likelihood = 1.0
        for i in range(self.n_features):
            likelihood *= self._calculate_likelihood(x[i], self.mean[c][i], self.variance[c][i])

        return likelihood * self.class_priors[c]

    def predict(self, X):
        """
        Predict class labels for samples in X

        Args:
            X: Test samples

        Returns:
            Predicted class labels
        """
        y_pred = []

        for x in X:
            posteriors = []
            for c in self.classes:
                posterior = self._calculate_class_probability(x, c)
                posteriors.append(posterior)

            y_pred.append(self.classes[np.argmax(posteriors)])

        return np.array(y_pred)

# Generate a sample dataset for demonstration
def create_sample_dataset():
    # We'll create a dataset related to weather and play tennis example
    # Features: Outlook (Sunny=0, Overcast=1, Rain=2), Temperature, Humidity, Windy (No=0, Yes=1)
    # Label: Play Tennis (No=0, Yes=1)

    data = {
        'Outlook': [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2],
        'Temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
        'Humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 80],
        'Windy': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
        'Play': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    }

    df = pd.DataFrame(data)
    df.to_csv('tennis_data.csv', index=False)
    return df

# Main execution
if __name__ == "__main__":
    # Create or load dataset
    try:
        df = pd.read_csv('tennis_data.csv')
        print("Dataset loaded from CSV file.")
    except FileNotFoundError:
        df = create_sample_dataset()
        print("Dataset created and saved to CSV file.")

    print("\nDataset Preview:")
    print(df.head())

    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train the classifier
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    # Make predictions
    y_pred = nb.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    target_names = ['No', 'Yes']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Display detailed predictions
    print("\nDetailed Predictions:")
    print("Sample\tActual\tPredicted")
    print("-" * 30)
    for i in range(len(y_test)):
        print(f"{i}\t{target_names[y_test[i]]}\t{target_names[y_pred[i]]}")

    # Create a new test instance
    print("\nPredict for a new instance:")
    new_instance = np.array([[0, 72, 80, 1]])  # Sunny, 72F, 80% humidity, Windy
    new_prediction = nb.predict(new_instance)
    print(f"New instance: Outlook=Sunny, Temp=72F, Humidity=80%, Windy=Yes")
    print(f"Prediction: {target_names[new_prediction[0]]}")

    # Demonstrate with a different dataset (Iris)
    print("\n\nDemonstration with Iris Dataset:")
    from sklearn import datasets

    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # Split data
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )

    # Train and evaluate
    nb_iris = NaiveBayesClassifier()
    nb_iris.fit(X_train_iris, y_train_iris)
    y_pred_iris = nb_iris.predict(X_test_iris)

    accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
    print(f"Accuracy on Iris dataset: {accuracy_iris*100:.2f}%")

    print("\nClassification Report for Iris:")
    print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
