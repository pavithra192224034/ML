def find_s(training_data):
    # Initialize the most specific hypothesis to the first positive example
    hypothesis = None
    for example in training_data:
        # If the example is positive
        if example[-1] == 'Yes':
            # Initialize hypothesis with the first positive example
            if hypothesis is None:
                hypothesis = example[:-1]
            else:
                # Update hypothesis to be the most specific generalization
                for i in range(len(hypothesis)):
                    if hypothesis[i] != example[i]:
                        hypothesis[i] = '?'
    return hypothesis

# Training data: Each example is a list of attribute values followed by the class label
# For example: [Attribute1, Attribute2, ..., AttributeN, ClassLabel]
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Run the FIND-S algorithm on the training data
most_specific_hypothesis = find_s(training_data)

# Print the most specific hypothesis
print("Most specific hypothesis:", most_specific_hypothesis)
