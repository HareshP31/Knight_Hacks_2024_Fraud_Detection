import streamlit as st
import pandas as pd
import os

# Define the temporary directory name
temp_dir = "temp"

# Check if the directory exists, if not create it
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Streamlit app title
st.title('Financial Fraud Detector!')

# Logic for uploading files, processing data, etc.
user_csv = st.file_uploader("Upload your CSV file here", type="csv")

if user_csv is not None:
    file_path = os.path.join(temp_dir, user_csv.name)  # Save to the temp directory
    with open(file_path, "wb") as f:
        f.write(user_csv.getbuffer())
    st.success(f"File '{user_csv.name}' uploaded and saved to '{temp_dir}'!")

    # Now read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    st.write("Uploaded Data:")
    st.dataframe(df)
