import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from fraud_detection_model import run_model
from fraud_detection_model import generate_spending_data_for_month

# Initialize session state to store uploaded data and other variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""


temp_dir = "temp"

st.title('Financial Fraud Detector!')

with st.sidebar:
    st.write('*Welcome to your personal Financial Fraud Detector*')

    st.caption('''The financial fraud detector website enables users to upload transaction spreadsheets for analysis to identify potential fraud. 
                  It validates data, uses machine learning algorithms to detect suspicious patterns, and assigns risk scores to highlight concerning transactions. 
                  The platform generates reports with visualizations to illustrate trends and offers recommendations for enhancing security, 
                  helping users proactively prevent financial fraud and build trust in their transactions.''')

    st.divider()

    st.caption("<p style ='text-align:center'>Made with love</p>", unsafe_allow_html=True)

# Function to plot line graph
def plot_line_graph(df, x_col, y_col, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.title(f'{file_name} - {y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    st.pyplot(plt)

# Use custom CSS to make the buttons bigger
# st.markdown("""
#     <style>
#     .stButton button {
#         width: 100%;
#         height: 60px;
#         font-size: 20px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Add buttons in each column


# Save directory
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Create two columns with equal width
col1, col2 = st.columns([1, 1])


with col1:
    button1 = st.button('Generate')

    if button1:
        flagged_df = run_model() 

        # Display the full DataFrame in Streamlit
        #st.subheader("Full Transactions Data")
        #st.dataframe(entire_df)

        # Display the transactions to review
        st.subheader("Transactions to Review")
        st.dataframe(flagged_df)

user_csv = st.file_uploader("**Upload your own CSV file here**", type="csv")

if user_csv is not None:

    # Define the temporary directory name
    temp_dir = "temp"

    # Check if the directory exists, if not create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if user_csv is not None:
        file_path = os.path.join(temp_dir, user_csv.name)  # Save to the temp directory
        with open(file_path, "wb") as f:
            f.write(user_csv.getbuffer())
        st.success(f"File '{user_csv.name}' uploaded and saved to '{temp_dir}'!")

        # Now read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        st.write("Uploaded Data:")
        st.dataframe(df)