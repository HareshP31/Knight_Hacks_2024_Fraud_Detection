import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib
from generate_csv import generate_spending_data_for_month  # IMPORT ALGORITHM

def run_model(df=None):
    # Load CSV file if not provided
    if df is None:
        try:
            csv_file = generate_spending_data_for_month(2024, 10)  
            df = pd.read_csv(csv_file)
            print("File loaded successfully!")
        except FileNotFoundError:
            print("Error: transactions.csv file not found. Please check the file path.")
            exit()  # Exit if the file is not found

    # Clean up column names
    df.columns = df.columns.str.strip()

    # Handle missing values
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
        if category == 'Suspicious':
            return 1.0  # High risk for suspicious category

        # Calculate risk score based on amount being outside of the range
        if amount < lower_bound:
            return (lower_bound - amount) / lower_bound  # Below range
        elif amount > upper_bound:
            return (amount - upper_bound) / amount  # Above range
        return 0.0  # Within normal range

    # Apply risk score calculation
    df['risk_score'] = df.apply(calculate_risk_score, axis=1)

    # Identify transactions to review
    to_review = df[(df['risk_score'] > 0.7) | (df['Category'] == 'Suspicious')]

    # One hot encoding of relevant columns   
    encoder = OneHotEncoder()  
    encoded_columns = encoder.fit_transform(df[['Category', 'MerchantType', 'Location']])
    encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(['Category', 'MerchantType', 'Location']))

    # Concatenate the encoded categories and drop the original columns
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=['Category', 'MerchantType', 'Location'], inplace=True)

    # Split the data into features (X) and target variable (y)
    X = df.drop(columns=['TransactionID', 'IsFraud', 'Timestamp'])  # Exclude unnecessary columns
    y = df['IsFraud']  # Target variable

    # Load or create the model
    try:
        model = joblib.load('fraud_detection_model.joblib')
        print("Model loaded.")
    except FileNotFoundError:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("No existing model found, creating a new one.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    joblib.dump(model, 'fraud_detection_model.joblib')

    # Make predictions
    predicted_probabilities = model.predict_proba(X_test)

    # Prepare the df_test DataFrame to store risk scores
    df_test = df.loc[X_test.index].copy()
    df_test['predicted_prob'] = predicted_probabilities[:, 1]  # Probability of fraud

    # Save transactions to review
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

    # Display feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(feature_importance_df.sort_values(by='Importance', ascending=False))    

    return df, to_review
