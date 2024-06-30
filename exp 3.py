import pandas as pd
import numpy as np
from collections import Counter
from math import log2

def entropy(data):
    labels = data.iloc[:, -1]
    label_counts = Counter(labels)
    total_count = len(labels)
    ent = -sum((count / total_count) * log2(count / total_count) for count in label_counts.values())
    return ent

def information_gain(data, split_attribute):
    total_entropy = entropy(data)
    values = data[split_attribute].unique()
    weighted_entropy = sum(
        (len(subset) / len(data)) * entropy(subset)
        for value in values
        for subset in [data[data[split_attribute] == value]]
    )
    return total_entropy - weighted_entropy

def id3(data, attributes, target_attribute):
    labels = data[target_attribute]
    
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    
    if len(attributes) == 0:
        return labels.mode()[0]
    
    best_attr = max(attributes, key=lambda attr: information_gain(data, attr))
    tree = {best_attr: {}}
    
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        new_attrs = [attr for attr in attributes if attr != best_attr]
        subtree = id3(subset, new_attrs, target_attribute)
        tree[best_attr][value] = subtree
        
    return tree

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    
    attr = next(iter(tree))
    value = sample[attr]
    subtree = tree[attr].get(value, "Unknown")
    return classify(subtree, sample)

# Load the dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Build the decision tree
attributes = list(data.columns[:-1])
target_attribute = 'PlayTennis'
decision_tree = id3(data, attributes, target_attribute)

# Print the decision tree
print("Decision Tree:")
print(decision_tree)

# Classify a new sample
new_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
prediction = classify(decision_tree, new_sample)
print("\nNew Sample Classification:")
print(f"Sample: {new_sample}")
print(f"Prediction: {prediction}")
