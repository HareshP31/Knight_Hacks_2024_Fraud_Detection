import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fraud_detection_model import csv_file 
import toml

# #Function to load the config from .toml file
# def load_config(file_path):
#     try:
#         config = toml.load(file_path)
#         return config
#     except FileNotFoundError:
#         st.error("Config file not found!")
#         return None
#     except toml.TomlDecodeError:
#         st.error("Error decoding the config file!")
#         return None

# # # Load configuration
# config = load_config("config.toml")

# Custom CSS styling
st.markdown("""
            
    <style>
     body {
         background-color: #FFFFFF;
     }
    /* File uploader styles */
    </style>
            
     """, unsafe_allow_html=True)


# Initialize session state to store uploaded data and other variables
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""

temp_dir = "temp"

st.title('Financial Fraud Detector:money_with_wings:')

st.write('**Click Generate to Trigger Machine Learning with our Personal Dataset:smile:**')

with st.sidebar:
    st.markdown("<h4 style='color: #007BFF;'>Welcome to your personal Financial Fraud Detector</h4>", unsafe_allow_html=True)

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

st.markdown("""
    <style>
    .stButton button {
        width: 80%;
        height: 50px;
        font-size: 20px;
        display: flex;
        justify-content: center;
        background-color: #007BFF; /* Your preferred blue color */
        color: white; /* Text color */
        transition: background-color 0.3s; /* Smooth transition for hover effect */
    }
    .stButton button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    </style>
""", unsafe_allow_html=True)


# Function to automatically load a file
def auto_upload_file(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)  # Adjust if you need to handle other file types
    else:
        st.error("File not found!")
        return None
    
# Save directory
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Create two columns with equal width
col1, col2 = st.columns([1, 1])


with col1:
    button1 = st.button('Generate')

predefined_file_path = "transactions.csv"

# Logic for button clicks
if button1:
    dataf = auto_upload_file(predefined_file_path)

    if dataf is not None:
        st.success(f"File '{predefined_file_path}' uploaded successfully!")
        st.dataframe(dataf)

st.write('**You Can Also Upload Your Own Monthly Statement**')
user_csv = st.file_uploader("", type="csv", key="file_uploader")

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

#     if user_csv is not None:

#         # Get the file name and save the file
#         st.session_state.file_name = user_csv.name
#         file_path = os.path.join("temp", st.session_state.file_name)

#         with open(file_path, "wb") as f:
#             f.write(user_csv.getbuffer())

#         st.success(f"File '{st.session_state.file_name}' uploaded and saved!")

#         # Read the CSV file into a DataFrame and store it in session state
#         st.session_state.uploaded_data = pd.read_csv(file_path)

# # Display the DataFrame if it exists in session state
# if st.session_state.uploaded_data is not None:
#     st.write("Uploaded Data:")
#     st.dataframe(st.session_state.uploaded_data)

#     # Perform analysis immediately after upload
#     if not st.session_state.uploaded_data.empty:
#         st.write("Performing analysis...")

#         # Example analysis: Calculate summary statistics
#         summary = st.session_state.uploaded_data.describe()
#         st.write("Summary Statistics:")
#         st.dataframe(summary)


# # Proceed to select columns for graphing
# st.write("Select columns for plotting:")
# x_col = st.selectbox("Select X-axis column:", st.session_state.uploaded_data.columns)
# y_col = st.selectbox("Select Y-axis column:", st.session_state.uploaded_data.columns)

# # Plot the line graph
# if st.button("Plot Line Graph"):
#     plot_line_graph(st.session_state.uploaded_data, x_col, y_col, st.session_state.file_name)

# # Additional analysis can be added here
# if st.button("Analyze Further"):
#     st.write("Further analysis could be implemented here.")
