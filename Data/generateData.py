import pandas as pd
import numpy as np
from faker import Faker
from datetime import timedelta, datetime

fake = Faker()

# Define spending limits for categories
spending_limits = {
    'Housing': (500, 800),
    'Subscriptions': (10, 20),
    'Food': (10, 30),
    'Gas/Transportation': (50, 70),
    'Clothing/Personal': (30, 60),
    'Miscellaneous': (20, 50),
    'Suspicious': (800, 2000)
}

# Fixed prices for certain categories
fixed_prices = {
    'Gym Membership': 29.99
}

# Tiered pricing for other merchants
tiered_pricing = {
    'Amazon Prime': [14.99, 7.49],
    'Netflix': [6.99, 15.49, 19.99],
    'Spotify': [9.99, 12.99, 15.99, 4.99]
}

merchants_by_category = {
    'Housing': ['Rent'],
    'Subscriptions': ['Netflix', 'Spotify', 'Amazon Prime', 'Gym Membership'],
    'Food': ['Walmart', 'Trader Joe\'s', 'Local Deli', 'Restaurants'],
    'Gas/Transportation': ['Shell', 'Uber', 'Gas Station', 'Lyft'],
    'Clothing/Personal': ['Gap', 'Sephora', 'Nike', 'H&M'],
    'Miscellaneous': ['Gift Shop', 'Charity', 'Lottery', 'General Store'],
    'Suspicious': ['Night Club', 'Casino', 'Luxury Goods', 'Jewelry Store']
}

# Function to generate spending data for a month
def generate_spending_data_for_month(year, month):
    categories = list(spending_limits.keys())
    
    # Define limits for how many times each category can appear
    category_limits = {
        'Housing': (1, 1),  # Only once
        'Subscriptions': (0, 2),  # Up to 2 times
        'Food': (3, 6),  # Up to 6 times
        'Gas/Transportation': (1, 2),  # Up to 2 times
        'Clothing/Personal': (1, 3),  # Up to 3 times
        'Miscellaneous': (1, 3),  # Up to 3 times
        'Suspicious': (0, 2)  # Up to 2 times
    }

    all_data = []  # List to hold all transactions
    total_transactions = 30
    rent_added = False  # Flag to track if Rent has been added

    # Create a list of dates for the specified month
    num_days = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days if month < 12 else (datetime(year + 1, 1, 1) - datetime(year, month, 1)).days
    dates = [datetime(year, month, day + 1).date() for day in range(num_days)]

    # Ensure Rent transaction is added on the first day
    first_day = dates[0]
    
    # Add Rent transaction
    amount = round(np.random.uniform(*spending_limits['Housing']), 2)
    transaction_data = {
        'TransactionID': len(all_data) + 1,
        'Category': 'Housing',
        'TransactionAmount': amount,
        'Location': 'Orlando',  # Fixed location
        'MerchantType': 'Rent',
        'IsFraud': 0,  # Not marked as fraud
        'Timestamp': first_day
    }
    all_data.append(transaction_data)  # Add Rent transaction
    rent_added = True  # Update flag

    # Track how many times each category has been used
    used_counts = {category: 0 for category in categories}
    used_counts['Housing'] += 1  # Increment for Rent

    # Get remaining unique dates to fill with transactions
    remaining_dates = dates[1:]  # Exclude the first day where Rent is added
    np.random.shuffle(remaining_dates)  # Shuffle remaining dates to randomize transaction assignment

    # Generate transactions for the remaining unique dates
    for current_date in remaining_dates[:total_transactions - 1]:  # We already added one transaction (Rent)
        category = np.random.choice(categories)

        # Ensure the category limit is respected
        if used_counts[category] < category_limits[category][1]:  # If below max limit
            # Allow other categories but only one Rent transaction
            amount = round(np.random.uniform(*spending_limits[category]), 2)
            merchant = np.random.choice(merchants_by_category[category])

            # Create transaction data
            transaction_data = {
                'TransactionID': len(all_data) + 1,
                'Category': category,
                'TransactionAmount': amount,
                'Location': 'Orlando',  # Fixed location
                'MerchantType': merchant,
                'IsFraud': 1 if category == 'Suspicious' else 0,  # Mark as fraud if suspicious
                'Timestamp': current_date + timedelta(hours=np.random.randint(0, 24), 
                                                      minutes=np.random.randint(0, 60))
            }
            all_data.append(transaction_data)  # Add to all data
            used_counts[category] += 1  # Increment used count for the category

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Sort the DataFrame by the Timestamp to ensure chronological order
    df.sort_values(by='Timestamp', inplace=True)
    
    return df

# Example usage: Generate data for October 2024
df = generate_spending_data_for_month(2024, 10)

# Display the first few rows
pd.set_option('display.max_rows', 30)
print(df)

df.to_csv('transactions_month.csv', index=False)