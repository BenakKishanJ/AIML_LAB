# Question 3: Develop a program to perform Candidate Elimination Algorithm to get Consistent Version Space

import numpy as np
import pandas as pd

class CandidateElimination:
    def __init__(self):
        self.specific_hypothesis = None
        self.general_hypothesis = None

    def fit(self, X, y):
        """
        Implement the Candidate Elimination algorithm.

        Args:
            X: Training data features
            y: Training data labels
        """
        # Convert data to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)

        # Initialize specific hypothesis as the first positive example
        positive_examples = X[y == True]
        if len(positive_examples) == 0:
            print("No positive examples found")
            return

        self.specific_hypothesis = positive_examples[0].copy()

        # Initialize general hypothesis with the most general constraints
        n_features = X.shape[1]
        self.general_hypothesis = [['?' for _ in range(n_features)] for _ in range(n_features)]

        # Create boundary sets
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    self.general_hypothesis[i][j] = self.specific_hypothesis[i]
                else:
                    self.general_hypothesis[i][j] = '?'

        print("Initial Specific Hypothesis:", self.specific_hypothesis)
        print("Initial General Hypothesis:", self.general_hypothesis)

        # Iterate through training examples
        for i, instance in enumerate(X):
            # If the instance is positive
            if y[i]:
                # Update specific hypothesis
                for j in range(n_features):
                    if self.specific_hypothesis[j] != instance[j]:
                        self.specific_hypothesis[j] = '?'

                # Remove from general hypothesis any hypothesis inconsistent with instance
                general_hypothesis_copy = self.general_hypothesis.copy()
                for hypothesis in general_hypothesis_copy:
                    if not self._is_consistent(hypothesis, instance):
                        self.general_hypothesis.remove(hypothesis)

            # If the instance is negative
            else:
                # Update general hypothesis
                general_hypothesis_copy = self.general_hypothesis.copy()
                for hypothesis in general_hypothesis_copy:
                    if self._is_consistent(hypothesis, instance):
                        self.general_hypothesis.remove(hypothesis)
                        for j in range(n_features):
                            if instance[j] != self.specific_hypothesis[j] and self.specific_hypothesis[j] != '?':
                                new_hypothesis = hypothesis.copy()
                                new_hypothesis[j] = self.specific_hypothesis[j]
                                self.general_hypothesis.append(new_hypothesis)

            print(f"\nIteration {i+1}:")
            print("Specific Hypothesis:", self.specific_hypothesis)
            print("General Hypothesis:", self.general_hypothesis)

            # Remove any overly general hypothesis
            self._remove_more_general()
            # Remove any overly specific hypothesis
            self._remove_more_specific()

        print("\nFinal Version Space:")
        print("S (Specific Boundary):", self.specific_hypothesis)
        print("G (General Boundary):", self.general_hypothesis)

    def _is_consistent(self, hypothesis, instance):
        """Check if an instance is consistent with a hypothesis"""
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
                return False
        return True

    def _remove_more_general(self):
        """Remove hypotheses that are more general than another hypothesis"""
        general_hypothesis_copy = self.general_hypothesis.copy()
        for i, h1 in enumerate(general_hypothesis_copy):
            for h2 in general_hypothesis_copy[i+1:]:
                if self._is_more_general(h1, h2):
                    if h1 in self.general_hypothesis:
                        self.general_hypothesis.remove(h1)
                elif self._is_more_general(h2, h1):
                    if h2 in self.general_hypothesis:
                        self.general_hypothesis.remove(h2)

    def _remove_more_specific(self):
        """Remove hypotheses that are more specific than the specific boundary"""
        general_hypothesis_copy = self.general_hypothesis.copy()
        for hypothesis in general_hypothesis_copy:
            if self._is_more_specific(hypothesis, self.specific_hypothesis):
                if hypothesis in self.general_hypothesis:
                    self.general_hypothesis.remove(hypothesis)

    def _is_more_general(self, h1, h2):
        """Check if h1 is more general than h2"""
        for i in range(len(h1)):
            if h1[i] != '?' and h2[i] == '?':
                return False
            if h1[i] != '?' and h1[i] != h2[i]:
                return False
        return True

    def _is_more_specific(self, h1, h2):
        """Check if h1 is more specific than h2"""
        return self._is_more_general(h2, h1)

# Example usage
if __name__ == "__main__":
    # Example data from classic Enjoy Sport problem
    data = pd.DataFrame([
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', True],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', True],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', False],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', True]
    ], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

    # Extract features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Run the algorithm
    ce = CandidateElimination()
    ce.fit(X, y)
