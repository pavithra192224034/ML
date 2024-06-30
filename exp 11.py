import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv("E:\\machine learning\\data set\\breastcancer.csv")

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Encode categorical target variable
label_encoder = LabelEncoder()
data['credit_score'] = label_encoder.fit_transform(data['credit_score'])

# Separate features and target variable
X = data.drop('credit_score', axis=1)
y = data['credit_score']

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display detailed classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
