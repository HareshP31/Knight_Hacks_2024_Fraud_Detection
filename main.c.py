import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title('Financial Fraud Dectector!')


#st.subheader('hi')

with st.sidebar:
    st.write('*Weclome to your personal Financial Fraud Detector*')

    st.caption('''The financial fraud detector website enables users to upload transaction spreadsheets for analysis to identify potential fraud. It validates data, uses machine learning algorithms to detect suspicious patterns, and assigns risk scores to highlight concerning transactions. The platform generates reports with visualizations to illustrate trends and offers recommendations for enhancing security, helping users proactively prevent financial fraud and build trust in their transactions.''')

    st.divider()


    st.caption("<p style ='text-align:center'>Made with love</p>", unsafe_allow_html = True)


# File uploader for CSV files
user_csv = st.file_uploader("Upload your CSV file here", type="csv")


# Function to plot line graph
def plot_line_graph(df, x_col, y_col, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.title(f'{file_name} - {y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    st.pyplot(plt)



if user_csv is not None:
    try:
        # Read the CSV file
        file_name = user_csv.name
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

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please check its format.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
