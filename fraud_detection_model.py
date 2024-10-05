import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
try:
    df = pd.read_csv('transactions.csv')
    print("File loaded successfully!")
except FileNotFoundError:
    print("Error: transactions.csv file not found. Please check the file path.")
    exit()  # Exit if the file is not found

# Display the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())
df.columns = df.columns.str.strip()

# Handle missing values in the 'amount' column
df['amount'].fillna(df['amount'].mean(), inplace=True)

# One hot encoding of the 'Category' and 'MerchantType' columns   
encoder = OneHotEncoder()
category_encoded = encoder.fit_transform(df[['category']])
merchant_encoded = encoder.fit_transform(df[['MerchantType']])

# Create DataFrames from the encoded columns
category_df = pd.DataFrame(category_encoded.toarray(), columns=encoder.get_feature_names_out(['category']))
merchant_df = pd.DataFrame(merchant_encoded.toarray(), columns=encoder.get_feature_names_out(['MerchantType']))

# Concatenate the encoded categories and merchants back into the main DataFrame
df = pd.concat([df, category_df, merchant_df], axis=1).drop(columns=['category', 'MerchantType'])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['TransactionID', 'IsFraud', 'Timestamp'])  # Exclude unnecessary columns
y = df['IsFraud']  # Target variable: IsFraud

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and get predicted probabilities
predicted_probabilities = model.predict_proba(X_test)

# Prepare the df_test DataFrame to store risk scores
df_test = df.loc[X_test.index].copy()  # Ensure df_test is correctly created
df_test['risk_score'] = predicted_probabilities[:, 1]  # Assign the probability of fraud

# Identify transactions to review (risk_score > 0.5) and save to a new CSV file
to_review = df_test[df_test['risk_score'] > 0.5]
to_review.to_csv('transactions_to_review.csv', index=False)
print("Transactions to review saved to 'transactions_to_review.csv'.")

# Display the first few transactions with their risk scores
print("Transactions with risk scores:")
print(df_test[['TransactionID', 'TransactionAmount', 'IsFraud', 'risk_score']].head())

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
