"""
We may have to adjust the code to accouunt for instances in which non-suspicious
 transactions that are within the upper spending limits are taking place various 
 times a week (i.e. someone made $200 transactions at walmart 4 times that week )

 We will also need to adjust the spending limits for food because no one's spending $150 at 
 restaurants multiple times a week. 

 Df's  transaction IDs  still need to be organized numerically. However, the time stamps are in fact in order. HUGE discrepancy
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import timedelta

fake = Faker()

# Define spending limits for categories
spending_limits = {
    'Housing': (500, 800),
    'Subscriptions': (10, 50),
    'Food': (150, 300),
    'Gas/Transportation': (50, 150),
    'Clothing/Personal': (30, 100),
    'Miscellaneous': (20, 100),
    'Suspicious': (800, 2000)
}

fixed_prices = {
    'Gym Membership': 29.99
}

# Tiered pricing for other merchants
tiered_pricing = {
    'Amazon Prime': [14.99, 7.49],
    'Netflix': [6.99, 15.49, 19.99],
    'Spotify': [9.99, 12.99, 15.99, 4.99]
}

# Transaction volume limits for a month
volume_limit_30_days = (40, 100)

# Function to generate random timestamps within a fixed period
def generate_random_timestamps_fixed_period(start_date, num_transactions, days_span=30):
    end_date = start_date + timedelta(days=days_span)
    timestamps = []

    for _ in range(num_transactions):
        random_date = fake.date_time_between_dates(datetime_start=start_date, datetime_end=end_date)
        timestamps.append(random_date)
    
    return timestamps

# Function to detect fraud based on rules
def detect_fraud(df, num_transactions, time_frame):
    # High volume of transaction in month = fraud
    if num_transactions > volume_limit_30_days[1]:
        df['IsFraud'] = 1
    
    # High transaction amount from non-suspicious merchant = fraud
    for index, row in df.iterrows():
        category = row['Category']
        amount = row['TransactionAmount']

        # Suspicious = automatically fraud
        if category == 'Suspicious':
            df.at[index, 'IsFraud'] = 1

        # Large transaction = automatically fraud
        elif amount > spending_limits[category][1]:
            df.at[index, 'IsFraud'] = 1
    
    # Check for >1 transaction at the same merchant within 24 hours
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values(by=['MerchantType', 'Timestamp'], inplace=True)

    fraud_indices = []
    for merchant, group in df.groupby('MerchantType'):
        for i in range(len(group) - 1):
            current_transaction_time = group.iloc[i]['Timestamp']
            next_transaction_time = group.iloc[i + 1]['Timestamp']
            if (next_transaction_time - current_transaction_time).total_seconds() <= 86400:
                fraud_indices.append(group.index[i])
                fraud_indices.append(group.index[i + 1])

    df.loc[fraud_indices, 'IsFraud'] = 1

    # Check for multiple subscription attempts within the same time frame
    subscription_indices = []
    df['YearMonth'] = df['Timestamp'].dt.to_period('M')
    for merchant, group in df[df['Category'] == 'Subscriptions'].groupby('MerchantType'):
        for year_month, month_group in group.groupby('YearMonth'):
            if len(month_group) > 1:
                subscription_indices.extend(month_group.index)

    df.loc[subscription_indices, 'IsFraud'] = 1

    return df

# Function to generate spending data
def generate_spending_data(num_transactions, start_date):
    categories = list(spending_limits.keys())

    data = {
        'TransactionID': range(1, num_transactions + 1),
        'Category': [],
        'TransactionAmount': [],
        'Location': ['Orlando' for _ in range(num_transactions)],  # Fixed location (Orlando)
        'Timestamp': generate_random_timestamps_fixed_period(start_date, num_transactions, days_span=30),
        'MerchantType': [],
        'IsFraud': [0] * num_transactions
    }

    # Adding realistic merchant names based on categories
    merchants_by_category = {
        'Housing': ['RentCo', 'MortgageHub', 'Home Depot', 'FurnishNow'],
        'Subscriptions': ['Netflix', 'Spotify', 'Amazon Prime', 'Gym Membership'],
        'Food': ['Walmart', 'Trader Joe\'s', 'Local Deli', 'Restaurants'],
        'Gas/Transportation': ['Shell', 'Uber', 'Gas Station', 'Lyft'],
        'Clothing/Personal': ['Gap', 'Ulta', 'Nike', 'H&M'],
        'Miscellaneous': ['Gift Shop', 'Charity', 'Lottery', 'General Store'],
        'Suspicious': ['Night Club', 'Casino', 'Luxury Goods', 'Jewelry Store']
    }
    
    # Generate categories and amounts consistently
    for _ in range(num_transactions):
        # Choose a category
        category = np.random.choice(categories)
        data['Category'].append(category)

        # Choose a corresponding merchant
        merchant = np.random.choice(merchants_by_category[category])
        data['MerchantType'].append(merchant)

        if category == 'Subscriptions' and merchant in fixed_prices:
            amount = fixed_prices[merchant]
        elif category == 'Subscriptions' and merchant in tiered_pricing:
            amount = np.random.choice(tiered_pricing[merchant])
        else:
            # Generate an amount within the limit for the chosen category
            amount = round(np.random.uniform(*spending_limits[category]), 2)

        data['TransactionAmount'].append(amount)

    df = pd.DataFrame(data)

    # Apply fraud detection
    df = detect_fraud(df, num_transactions, 'monthly')

    # Sort the DataFrame by Timestamp again to ensure proper ordering
    df.sort_values(by='Timestamp', inplace=True)

    # Reset the TransactionID to maintain order after sorting
    df['TransactionID'] = range(1, len(df) + 1)

    return df

# Example usage: Generate data for 100 transactions
num_transactions = 100
start_date = fake.date_time_this_year()  # Starting date for the first transaction

df = generate_spending_data(num_transactions, start_date)

# Display the first few rows
print(df)

# Optionally, save the data to a CSV file
# df.to_csv('synthetic_transactions.csv', index=False)
