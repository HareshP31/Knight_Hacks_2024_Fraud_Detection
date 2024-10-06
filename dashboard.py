import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from fraud_detection_model import run_model
# from fraud_detection_model import generate_spending_data_for_month

# Initialize session state to store uploaded data and other variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""


temp_dir = "temp"

st.title('Financial Fraud Detector!')

with st.sidebar:
    st.write('**Welcome to your personal Financial Fraud Detector**')

    st.caption('''The financial fraud detector website enables users to upload transaction spreadsheets for analysis to identify potential fraud. 
                  It validates data, uses machine learning algorithms to detect suspicious patterns, and assigns risk scores to highlight concerning transactions. 
                  The platform generates reports with visualizations to illustrate trends and offers recommendations for enhancing security, 
                  helping users proactively prevent financial fraud and build trust in their transactions.''')

    st.divider()

    st.caption("<p style ='text-align:center'>Made with love</p>", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Styling for the sidebar */
    .st-emotion-cache-1gwvy71.eczjsme12 {
        background-color: #f0f0f5; /* Light gray background */
        padding: 20px; /* Add padding */
    }

    /* Text inside the sidebar */
    .st-emotion-cache-1gwvy71.eczjsme12 p, .st-emotion-cache-1gwvy71.eczjsme12 strong {
        color: #333333; /* Dark text color */
    }
    
    /* Styling for captions or centered text in the sidebar */
    .st-emotion-cache-1l5t55b p {
        text-align: center;
        font-weight: bold;
    }
    .stButton button {
        width: 80%;
        height: 50px;
        font-size: 20px;
        display: flex;
        justify-content: center;
        background-color: #007BFF; 
        color: white;
        transition: background-color 0.3s; /* Smooth transition for hover effect */
    }
    .stButton button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    </style>
""", unsafe_allow_html=True)

def display_dataframes(full_df, review_df):
    st.subheader("Full Transaction Data")
    st.dataframe(full_df)
    
    st.subheader("Transactions to Review")
    st.dataframe(review_df)

# Save directory
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Create two columns with equal width
col1, col2 = st.columns([1, 1])


with col1:
    if st.button('Generate'):
        full_df, flagged_df = run_model()  # Get both DataFrames

        if full_df is not None and flagged_df is not None:
            display_dataframes(full_df, flagged_df)
with col2:
    user_csv = st.file_uploader("**Upload your own CSV file here**", type=["csv"])

    if user_csv is not None:
        df = pd.read_csv(user_csv)
        df.columns = df.columns.str.strip()  # Strip whitespace from column names

        # Debugging: Print columns to verify
        # st.write("Columns in uploaded DataFrame:", df.columns.tolist())
        
        # Check for the 'TransactionAmount' column
        if 'TransactionAmount' not in df.columns:
            st.error("Error: 'TransactionAmount' column not found in the uploaded file.")
        else:
            full_df_uploaded, flagged_df_uploaded = run_model(df)  # Call your function with the uploaded DataFrame
            st.success('Transactions processed and saved!')
            display_dataframes(full_df_uploaded, flagged_df_uploaded)