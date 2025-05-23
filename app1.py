import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# Load the saved model and preprocessing objects
@st.cache_resource
def load_artifacts():
    model = joblib.load('fraud_detection_logreg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('expected_columns.pkl')
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# App title and description
st.title("Fraud Detection System")
st.write("""
This application helps detect potentially fraudulent financial transactions using machine learning.
""")

# Navigation
page = st.sidebar.selectbox("Choose a page", ["Welcome", "EDA", "Model Prediction"])

if page == "Welcome":
    st.header("Welcome to the Fraud Detection System")
    st.write("""
    ### About This Application
    
    This system uses a machine learning model to identify potentially fraudulent transactions based on:
    - Transaction characteristics
    - User behavior patterns
    - Historical transaction data
    
    ### How to Use
    1. **EDA Page**: Explore the dataset and understand transaction patterns
    2. **Model Prediction Page**: Input transaction details to get a fraud prediction
    
    ### Dataset Overview
    The model was trained on transaction data with the following features:
    - Transaction amount, type, time, and location
    - User account information
    - Transaction history patterns
    """)
    
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Load data with caching
    @st.cache_data
    def load_data():
        df = pd.read_csv("Fraud Detection Dataset.csv")
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].mean())
        
        # IQR-based outlier handling for EDA visualization
        Q1_amt = df['Transaction_Amount'].quantile(0.25)
        Q3_amt = df['Transaction_Amount'].quantile(0.75)
        IQR_amt = Q3_amt - Q1_amt
        upper_amt = Q3_amt + 1.5 * IQR_amt
        df['Transaction_Amount_clean'] = np.where(df['Transaction_Amount'] > upper_amt, 
                                                  upper_amt, 
                                                  df['Transaction_Amount'])
        
        Q1_trans = df['Number_of_Transactions_Last_24H'].quantile(0.25)
        Q3_trans = df['Number_of_Transactions_Last_24H'].quantile(0.75)
        IQR_trans = Q3_trans - Q1_trans
        upper_trans = Q3_trans + 1.5 * IQR_trans
        df['Number_of_Transactions_Last_24H_clean'] = np.where(
            df['Number_of_Transactions_Last_24H'] > upper_trans,
            upper_trans,
            df['Number_of_Transactions_Last_24H']
        )
        
        return df
    
    df = load_data()
    
    st.subheader("Dataset Sample")
    st.write(df.head())
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Outlier information
    st.subheader("Outlier Information")
    st.write(f"""
    - Transaction amounts above ${df['Transaction_Amount'].quantile(0.99):.2f} (99th percentile) are considered outliers
    - More than {df['Number_of_Transactions_Last_24H'].quantile(0.99):.1f} transactions in 24H is considered unusual
    """)
    
    # Fraud distribution
    st.subheader("Fraud Distribution")
    fraud_counts = df['Fraudulent'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(fraud_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%')
    st.pyplot(fig)
    
    # Transaction amount distribution (with outlier handling)
    st.subheader("Transaction Amount Distribution (Outliers Capped at 99th Percentile)")
    fig, ax = plt.subplots()
    sns.histplot(df['Transaction_Amount_clean'], bins=50, ax=ax)
    ax.set_xlabel("Transaction Amount (Outliers Capped)")
    st.pyplot(fig)
    
    # Boxplot of transaction amounts by fraud status (with outlier handling)
    st.subheader("Transaction Amount by Fraud Status (Outliers Removed)")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Fraudulent', y='Transaction_Amount_clean', 
                showfliers=False, ax=ax)
    st.pyplot(fig)
    
    # Transaction type analysis
    st.subheader("Transaction Type vs Fraud")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Transaction_Type', hue='Fraudulent', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Time of transaction analysis
    st.subheader("Time of Transaction vs Fraud")
    df['Hour'] = df['Time_of_Transaction'] % 24
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Fraudulent', y='Hour', ax=ax)
    st.pyplot(fig)
    
    # Feature Importance Boxplots
    st.subheader("Feature Distributions (Important Features)")
    
    # Select important features to visualize
    important_features = [
        'Transaction_Amount_clean',
        'Number_of_Transactions_Last_24H_clean',
        'Account_Age',
        'Hour',
        'Previous_Fraudulent_Transactions'
    ]
    
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(important_features), figsize=(20, 5))
    
    # Plot each feature's distribution
    for ax, col in zip(axes, important_features):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
        ax.set_ylabel('')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Model Prediction":
    st.header("Fraud Prediction")
    
    # Create input form
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
            time_of_transaction = st.number_input("Time of Transaction (hours since start)", min_value=0, value=12)
            account_age = st.number_input("Account Age (days)", min_value=0, value=100)
            num_transactions_24h = st.number_input("Number of Transactions in Last 24H", min_value=0, value=5)
            prev_fraud = st.number_input("Previous Fraudulent Transactions", min_value=0, max_value=4, value=0)
            
        with col2:
            transaction_type = st.selectbox("Transaction Type", 
                                          ["Purchase", "Withdrawal", "Transfer"])
            device_used = st.selectbox("Device Used", ["Mobile", "Desktop", "Tablet", "Unknown"])
            location = st.selectbox("Location", ["Domestic", "International"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Credit Card", "Debit Card", "Bank Transfer"])
        
        submitted = st.form_submit_button("Predict Fraud Risk")
    
    if submitted:
        # Create a dataframe from the input
        input_data = {
            'Transaction_Amount': [transaction_amount],
            'Time_of_Transaction': [time_of_transaction],
            'Account_Age': [account_age],
            'Number_of_Transactions_Last_24H': [num_transactions_24h],
            'Previous_Fraudulent_Transactions': [prev_fraud],
            'Transaction_Type': [transaction_type],
            'Device_Used': [device_used],
            'Location': [location],
            'Payment_Method': [payment_method]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Create features with outlier handling
        def create_features(input_df):
            # Cap extreme values for numerical features
            input_df['Transaction_Amount'] = np.minimum(input_df['Transaction_Amount'], 100000)  # Cap at $100,000
            input_df['Number_of_Transactions_Last_24H'] = np.minimum(input_df['Number_of_Transactions_Last_24H'], 50)  # Cap at 50
            
            # Time-based features
            input_df['Hour'] = input_df['Time_of_Transaction'] % 24
            input_df['Is_Night'] = ((input_df['Hour'] < 6) | (input_df['Hour'] > 20)).astype(int)
            input_df['Weekend'] = ((input_df['Time_of_Transaction'] // 24) % 7 >= 5).astype(int)
            
            # Transaction velocity features (with smoothing to avoid division by zero)
            input_df['Amount_per_Transaction'] = input_df['Transaction_Amount'] / (input_df['Number_of_Transactions_Last_24H'] + 1)
            input_df['Transaction_Velocity'] = input_df['Number_of_Transactions_Last_24H'] / (input_df['Account_Age'] + 7)  # +7 to smooth
            
            # Behavioral features with robust scaling
            median_amount = 2996.25
            iqr_amount = 5043.93
            input_df['Amount_Deviation'] = (input_df['Transaction_Amount'] - median_amount) / iqr_amount
            
            # Risk scoring with log transform for amount
            input_df['Risk_Score'] = (input_df['Previous_Fraudulent_Transactions'] * 
                                    np.log1p(input_df['Transaction_Amount']) * 
                                    (1 + np.log1p(input_df['Transaction_Velocity'])))
            
            # Interaction features
            input_df['Amount_Location_Interaction'] = np.log1p(input_df['Transaction_Amount']) * (input_df['Location'] == 'International').astype(int)
            
            # Percentile features - using robust estimates
            input_df['Amount_Percentile'] = np.where(
                input_df['Transaction_Amount'] < 1000, 0.3,
                np.where(input_df['Transaction_Amount'] < 5000, 0.6, 0.9)
            )
            
            input_df['24H_Transactions_Percentile'] = np.where(
                input_df['Number_of_Transactions_Last_24H'] < 5, 0.3,
                np.where(input_df['Number_of_Transactions_Last_24H'] < 15, 0.6, 0.9)
            )
            
            return input_df
        
        input_df = create_features(input_df)
        
        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df)
        
        # Ensure all expected columns are present (excluding 'Fraudulent')
        expected_features = [col for col in expected_columns if col != 'Fraudulent']
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[expected_features]
        
        # Scale the data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")
        
        st.write(f"Probability of being fraudulent: {prediction_proba[0][1]:.2%}")
        
        # Show feature importance (if available)
        if hasattr(model, 'coef_'):
            st.subheader("Top Contributing Factors")
            coefs = pd.Series(model.coef_[0], index=expected_features)
            top5 = coefs.abs().sort_values(ascending=False).head(5)
            
            for feature in top5.index:
                value = input_df[feature].values[0]
                impact = coefs[feature]
                direction = "increases" if impact > 0 else "decreases"
                st.write(f"- **{feature}**: Value = {value:.2f}, {direction} fraud risk (impact = {impact:.2f})")
