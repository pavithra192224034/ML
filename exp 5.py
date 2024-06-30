import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter

# Load the Iris dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(url, header=None, names=column_names)
    return data

# Standardize the dataset
def standardize_data(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std

# K-Nearest Neighbors algorithm
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Compute distances from test_point to all training points
        distances = [np.linalg.norm(test_point - train_point) for train_point in X_train]
        
        # Get the k nearest neighbors and their labels
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Majority vote for the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Main function
def main():
    # Load the data
    data = load_data()
    print("First few rows of the dataset:")
    print(data.head())

    # Separate features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Standardize the features
    X_std = standardize_data(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

    # Set the number of neighbors
    k = 3

    # Make predictions
    y_pred = knn_predict(X_train, y_train, X_test, k)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K-NN classification accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
