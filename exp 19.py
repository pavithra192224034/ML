import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# Assuming the dataset is in a CSV file named 'bank_loan_data.csv'
data = pd.read_csv("E:/machine learning/data set/Position_Salaries.csv")

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)

# Convert categorical features to numerical
label_encoders = {}
categorical_features = ['employment_status', 'loan_approved']

for column in categorical_features:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target variable
X = data.drop('loan_approved', axis=1)
y = data['loan_approved']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display detailed classification report
print(classification_report(y_test, y_pred))
