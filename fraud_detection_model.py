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

# Handle missing values in the 'TransactionAmount' column
df['TransactionAmount'].fillna(df['TransactionAmount'].mean(), inplace=True)

# Define price ranges for each merchant category
price_ranges = {
    'Housing': (500, 800),
    'Subscriptions': (10, 20),
    'Food': (10, 30),
    'Gas/Transportation': (50, 70),
    'Clothing/Personal': (30, 60),
    'Miscellaneous': (20, 50),
    'Suspicious': (800, 2000)
}

# Function to calculate risk score based on transaction amount and category
def calculate_risk_score(row):
    category = row['Category']
    amount = row['TransactionAmount']

    if category not in price_ranges:
        return 0.0  # Default to 0 if category is unknown
    
    lower_bound, upper_bound = price_ranges[category]
    
    # Check for suspicious category
    if category == 'Suspicious':
        return 1.0
    
    # Calculate risk score based on amount being outside of the range
    if amount < lower_bound:
        return (lower_bound - amount) / lower_bound  # Normalized score for being below range
    elif amount > upper_bound:
        return (amount - upper_bound) / amount  # Normalized score for being above range
    else:
        return 0.0  # Within normal range, score is 0

# Apply the risk score calculation
df['risk_score'] = df.apply(calculate_risk_score, axis=1)

to_review = df[(df['risk_score'] > 0.7) | (df['Category'] == 'Suspicious')]

# One hot encoding of the 'Category' and 'MerchantType' columns   
encoder = OneHotEncoder()  
encoded_columns = encoder.fit_transform(df[['Category', 'MerchantType', 'Location']])

# Create DataFrames from the encoded columns
encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(['Category', 'MerchantType', 'Location']))

# Concatenate the encoded categories and merchants back into the main DataFrame
df = pd.concat([df, encoded_df], axis=1)
df.drop(columns=['Category', 'MerchantType', 'Location'], inplace=True)


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
df_test['predicted_prob'] = predicted_probabilities[:, 1]  # Assign the probability of fraud

# Identify transactions to review (predicted_prob > 0.5) and save to a new CSV file

to_review.to_csv('transactions_to_review.csv', index=False)
print("Transactions to review saved to 'transactions_to_review.csv'.")

# Display the first few transactions with their risk scores
print("Transactions with risk scores:")
print(df_test[['TransactionID', 'TransactionAmount', 'IsFraud', 'risk_score', 'predicted_prob']].head())

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
