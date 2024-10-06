# Project: Risk Radar

## Project Description

The increasing number of fraudulent transactions in today's financial landscape motivated us to create a solution that would empower individuals and businesses to quickly identify and act on suspicious activities. Our aim was to create an accessible and efficient tool to help improve financial security and reduce fraud risks.

## Functionality

Uses machine learning to analyze transaction data and determine the likelihood of fraud
- Identifies potential fraudulent activities based on patterns within the data
- Provides a risk score for each transaction
- High-risk transactions are flagged for further review

## Features

- Generate button that creates a new csv file using our data generation algorithm based on the demographics we are looking for, then runs it through our model and displays the data
- Upload button that allows user to upload their own csv file to run with our model and display the data

## Technologies

- The project is built using Python and popular data science libraries.
- Utilizes pandas and numpy for data generation and processing.
- scikit-learn is used to train and evaluate a Random Forest model.
- The final model is deployed in a streamlined application using Streamlit for a user-friendly interface.
- Used joblib to store model intelligence
- Implemented faker for data generation

## Challenges

- There was a learning curve in understanding both the relevant libraries and Python's syntax.
- We encountered issues reformatting datasets to facilitate the fraud detection model’s learning.
- Handling missing values and encoding categorical features was particularly challenging.
- Finding appropriate data to train the model was difficult (to solve the data issue, we created a data generation algorithm based on the spending habits of an average college student using data from the Federal Reserve Bank of Atlanta)

## Streacth Goals

- Add more feature dectection (detect fraud based on location, time, etc.)
- Add more flexibility with user upload (covert a pdf to a csv file/ covert csv file to the format our model needs)
- Add more options to display the data in different ways
- Add a chatbot to advise user on what to do next or answer questions

**Clone the repository:**

   ```bash
   git clone https://github.com/HareshP31/Knight_Hacks_2024_Fraud_Detection
