import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Function to plot line graph
def plot_line_graph(df, x_col, y_col, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.title(f'{file_name} - {y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    st.pyplot(plt)


st.title('Financial Fraud Dectector!')


#st.subheader('hi')

with st.sidebar:
    st.write('*Weclome to your personal Financial Fraud Detector*')

    st.caption('''The financial fraud detector website enables users to upload transaction spreadsheets for analysis to identify potential fraud. It validates data, uses machine learning algorithms to detect suspicious patterns, and assigns risk scores to highlight concerning transactions. The platform generates reports with visualizations to illustrate trends and offers recommendations for enhancing security, helping users proactively prevent financial fraud and build trust in their transactions.''')

    st.divider()


    st.caption("<p style ='text-align:center'>Made with love</p>", unsafe_allow_html = True)



col1, col2 = st.columns([1, 1])

# Use custom CSS to make the buttons bigger
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)



# Add buttons in each column
with col1:
    button1 = st.button('Generate')
with col2:
    button2 = st.button('Upload')

# Logic for button clicks
if button1:
    st.write("Click Proceed!")
    st.button('Proceed')

if button2:
    user_csv = st.file_uploader("Upload your CSV file here", type="csv")
    # File uploader for CSV files


    if user_csv is not None:
        
        # Read the CSV file
        file_name = user_csv.name

        # Save the uploaded file to a temporary directory
        with open(os.path.join("temp", file_name), "wb") as f:
            f.write(user_csv.getbuffer())

        st.success(f"File '{file_name}' uploaded and saved!")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join("temp", file_name))

        df = pd.read_csv(user_csv)
            

        # Display the DataFrame
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Select columns for the x and y axes
        st.write("Select columns for X and Y axes:")
        x_col = st.selectbox("Select X-axis column:", df.columns)
        y_col = st.selectbox("Select Y-axis column:", df.columns)

        # Plot the line graph
        if st.button("Plot Line Graph"):
            plot_line_graph(df, x_col, y_col, file_name)

