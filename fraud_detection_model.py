import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

try:
    df = pd.read_csv('transactions.csv')
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: transactions.csv file not found. Please check the file path.")
# one hot encoding of the category column
encoder = OneHotEncoder()
category_encoded = encoder.fit_transform(df[['category']]).toarray()
# categories to dataframe
category_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['category']))

# Concatenate the encoded categories back into the main DataFrame
df = pd.concat([df, category_df], axis=1)

# Drop the original 'category' column
df = df.drop(columns=['category'])

# Split the data into features and target variable
X = df.drop(columns=['transaction_id', 'is_fraud'])  # Exclude transaction_id and target
y = df['is_fraud']  # Target variable: is_fraud

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(df.head())

print(df.isnull().sum())

print(df.describe())

df['amount'].fillna(df['amount'].mean(), inplace=True)

X = df[['amount']]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Optionally, save the model
import joblib
joblib.dump(model, 'fraud_detection_model_with_categories.pkl')