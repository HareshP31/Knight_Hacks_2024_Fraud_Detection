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
    # Define limits for suspicious transactions
    'Suspicious': (800, 2000)
}

fixed_prices = {
    'Gym Membership': 29.99

}
#Tiered pricing for other merchants
#
tiered_pricing = {
    'Amazon Prime': [14.99, 7.49],
    'Netflix' : [6.99, 15.49, 19.99],
    'Spotify' : [9.99,12.99,15.99,4.99]
}

#Transaction volume limits for a month
volume_limit_30_days = (40,100)

#def generate_num_transactions_30_days


# Function to generate spending data
def generate_spending_data(num_transactions, start_date):
    categories = list(spending_limits.keys())

    data = {
        'TransactionID': range(1, num_transactions + 1),
        'Category': [],
        'TransactionAmount': [],
        'Location': ['Orlando' for _ in range(num_transactions)],  # Fixed location (Orlando)
        #'Timestamp':generate_num_transactions_30_days
        'MerchantType': [],
        'IsFraud': []
    }

    # Adding realistic merchant names based on categories
    merchants_by_category = {
        'Housing': ['RentCo', 'MortgageHub', 'Home Depot', 'FurnishNow'],
        'Subscriptions': ['Netflix', 'Spotify', 'Amazon Prime', 'Gym Membership'],
        'Food': ['Walmart', 'Trader Joe\'s', 'Local Deli', 'Restaurants'],
        'Gas/Transportation': ['Shell', 'Uber', 'Gas Station', 'Lyft'],
        'Clothing/Personal': ['Gap', 'Sephora', 'Nike', 'H&M'],
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
        data['isFraud'].append(0)

    df = pd.DataFrame(data)

 
    

# Example usage: Generate data for 100 transactions
num_transactions = 100
start_date = fake.date_time_this_year()  # Starting date for the first transaction

df = generate_spending_data(num_transactions, start_date)

# Display the first few rows
print(df.head())

# df.to_csv('synthetic_transactions.csv', index=False)

