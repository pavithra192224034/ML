import pandas as pd


def load_data(filename):
    data = pd.read_csv(filename)
    return data


def is_more_specific(h1, h2):
    for x, y in zip(h1, h2):
        if x != '?' and x != y:
            return False
    return True

def candidate_elimination(data):
    
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    print("First few rows of the dataset:")
    print(data.head())

    
    specific_h = None
    for i, example in enumerate(features):
        if target[i] == 'Yes':  
            specific_h = example
            break

    if specific_h is None:
        raise ValueError("No positive examples in the dataset.")

    
    general_h = [['?' for _ in range(len(specific_h))]]

    print(f"Initial specific hypothesis: {specific_h}")
    print(f"Initial general hypothesis: {general_h}")

    
    for i, example in enumerate(features):
        if target[i] == 'Yes':
            # Update the specific boundary
            for j in range(len(specific_h)):
                if example[j] != specific_h[j]:
                    specific_h[j] = '?'
            
            
            general_h = [g for g in general_h if is_more_specific(g, example)]
        else:
           
            general_h_new = []
            for g in general_h:
                if is_more_specific(g, example):
                    for j in range(len(g)):
                        if g[j] == '?':
                            new_hypothesis = g.copy()
                            new_hypothesis[j] = example[j]
                            general_h_new.append(new_hypothesis)
            general_h.extend(general_h_new)
            general_h = [g for g in general_h if is_more_specific(g, specific_h)]
    
    return specific_h, general_h


def print_hypotheses(specific_h, general_h):
    print("Most specific hypothesis (S):")
    print(specific_h)
    print("\nMost general hypotheses (G):")
    for hypothesis in general_h:
        print(hypothesis)


filename = "E:\\machine learning\\data set\\breastcancer.csv"
data = load_data(filename)
specific_h, general_h = candidate_elimination(data)
print_hypotheses(specific_h, general_h)
