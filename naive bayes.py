import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(url, header=None, names=column_names)
    return data


def main():
      
    data = load_data()
    print("First few rows of the dataset:")
    print(data.head())

    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

   
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

    gnb = GaussianNB()

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naïve Bayes classification accuracy: {accuracy * 100:.2f}%")

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
