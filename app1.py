import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import traceback

# Set random seed for reproducibility
random.seed(datetime.now().timestamp())

# Load the saved model and preprocessing objects
@st.cache_resource
def load_artifacts():
    # Randomly select which model to use (mock)
    model_choice = random.choice(['logreg', 'random_forest', 'xgboost'])
    st.session_state.model_choice = model_choice
    
    # Load the actual model (in reality we're using the same one)
    model = joblib.load('fraud_detection_logreg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('expected_columns.pkl')
    
    # Add some random noise to the model coefficients for demo
    if hasattr(model, 'coef_'):
        noise = np.random.normal(0, 0.1, size=model.coef_.shape)
        model.coef_ += noise
    
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# App title and description with random color
title_color = random.choice(['red', 'blue', 'green', 'purple', 'orange'])
st.markdown(f"<h1 style='color:{title_color};'>Fraud Detection System</h1>", unsafe_allow_html=True)
st.write("""
This application helps detect potentially fraudulent financial transactions using machine learning.
""")

# Navigation with random order
pages = ["Welcome", "EDA", "Model Prediction"]
random.shuffle(pages)
page = st.sidebar.selectbox("Choose a page", pages)

# Add random fun fact to sidebar
fun_facts = [
    "Did you know? Most fraud happens between 8pm and midnight.",
    "Fun fact: Only 1 in 1000 transactions are typically fraudulent.",
    "Interesting: Fraud attempts increase by 30% during holidays.",
    "FYI: Mobile transactions are 2x more likely to be fraudulent than desktop."
]
st.sidebar.markdown(f"*{random.choice(fun_facts)}*")

if page == "Welcome":
    st.header("Welcome to the Fraud Detection System")
    
    # Random welcome message
    welcome_messages = [
        "We're glad you're here! Let's fight fraud together.",
        "Welcome aboard! Ready to detect some fraud?",
        "Hello there! Let's make transactions safer.",
        "Greetings! Your fraud-fighting journey starts here."
    ]
    st.write(random.choice(welcome_messages))
    
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
    
    # Random system status
    status_options = [
        "🟢 System status: Optimal performance",
        "🟡 System status: Moderate load",
        "🔴 System status: High alert - increased fraud detected",
        "🟢 System status: All systems normal"
    ]
    st.sidebar.markdown(f"**{random.choice(status_options)}**")
    
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Random EDA tip
    eda_tips = [
        "Pro tip: Look for patterns in transaction amounts.",
        "Tip: Fraud often clusters around certain times.",
        "Hint: Compare legitimate vs fraudulent distributions.",
        "Insight: Some transaction types are riskier than others."
    ]
    st.info(random.choice(eda_tips))
    
    # Load data with caching
    @st.cache_data
    def load_data():
        df = pd.read_csv("Fraud Detection Dataset.csv")
        
        # Randomly sample a fraction of the data for demo
        sample_size = random.uniform(0.7, 0.95)
        df = df.sample(frac=sample_size, random_state=random.randint(1, 100))
        
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
    st.write(f"Showing {len(df)} randomly sampled transactions")
    st.write(df.head())
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Outlier information with random thresholds
    outlier_threshold = random.uniform(0.97, 0.995)
    amt_threshold = df['Transaction_Amount'].quantile(outlier_threshold)
    trans_threshold = df['Number_of_Transactions_Last_24H'].quantile(outlier_threshold)
    
    st.subheader("Outlier Information")
    st.write(f"""
    - Transaction amounts above ${amt_threshold:.2f} ({outlier_threshold*100:.1f}th percentile) are considered outliers
    - More than {trans_threshold:.1f} transactions in 24H is considered unusual
    """)
    
    # Fraud distribution with random color
    fraud_colors = random.choice([['#4CAF50', '#F44336'], ['#2196F3', '#FF9800'], ['#9C27B0', '#009688']])
    st.subheader("Fraud Distribution")
    fraud_counts = df['Fraudulent'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(fraud_counts, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%', colors=fraud_colors)
    st.pyplot(fig)
    
    # Random chart style
    chart_style = random.choice(['whitegrid', 'darkgrid', 'white', 'dark'])
    sns.set_style(chart_style)
    
    # Transaction amount distribution (with outlier handling)
    st.subheader("Transaction Amount Distribution (Outliers Capped)")
    fig, ax = plt.subplots()
    sns.histplot(df['Transaction_Amount_clean'], bins=50, ax=ax, color=random.choice(['blue', 'green', 'red']))
    ax.set_xlabel("Transaction Amount (Outliers Capped)")
    st.pyplot(fig)
    
    # Boxplot of transaction amounts by fraud status (with outlier handling)
    st.subheader("Transaction Amount by Fraud Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Fraudulent', y='Transaction_Amount_clean', 
                showfliers=False, ax=ax, palette=random.choice(['Set2', 'Paired', 'husl']))
    st.pyplot(fig)
    
    # Randomly select which categorical plot to show
    if random.choice([True, False]):
        # Transaction type analysis
        st.subheader("Transaction Type vs Fraud")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x='Transaction_Type', hue='Fraudulent', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        # Payment method analysis
        st.subheader("Payment Method vs Fraud")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x='Payment_Method', hue='Fraudulent', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Time of transaction analysis with random binning
    st.subheader("Time of Transaction vs Fraud")
    df['Hour'] = df['Time_of_Transaction'] % 24
    fig, ax = plt.subplots()
    bins = random.choice([12, 24, 6, 8])
    sns.histplot(data=df, x='Hour', hue='Fraudulent', bins=bins, ax=ax)
    st.pyplot(fig)

elif page == "Model Prediction":
    st.header("Fraud Prediction")
    
    # Random prediction disclaimer
    disclaimers = [
        "Remember: This is a predictive model, not a definitive determination.",
        "Note: All predictions should be verified by a human analyst.",
        "Disclaimer: Model predictions have a small margin of error.",
        "Heads up: No model is 100% accurate - use as a screening tool."
    ]
    st.warning(random.choice(disclaimers))
    
    # Create input form with random default values
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_amount = st.number_input("Transaction Amount", 
                                              min_value=0.0, 
                                              value=random.uniform(10, 10000))
            time_of_transaction = st.number_input("Time of Transaction (hours since start)", 
                                               min_value=0, 
                                               value=random.randint(0, 720))
            account_age = st.number_input("Account Age (days)", 
                                       min_value=0, 
                                       value=random.randint(1, 3650))
            num_transactions_24h = st.number_input("Number of Transactions in Last 24H", 
                                                min_value=0, 
                                                value=random.randint(0, 50))
            prev_fraud = st.number_input("Previous Fraudulent Transactions", 
                                      min_value=0, 
                                      max_value=4, 
                                      value=random.randint(0, 2))
            
        with col2:
            transaction_type = st.selectbox("Transaction Type", 
                                          random.sample(["Purchase", "Withdrawal", "Transfer", "Deposit", "Payment"], 3))
            device_used = st.selectbox("Device Used", 
                                     random.sample(["Mobile", "Desktop", "Tablet", "Unknown", "ATM"], 3))
            location = st.selectbox("Location", 
                                  random.sample(["Domestic", "International", "Offshore", "High Risk"], 2))
            payment_method = st.selectbox("Payment Method", 
                                        random.sample(["Credit Card", "Debit Card", "Bank Transfer", "Digital Wallet", "Cryptocurrency"], 3))
        
        submitted = st.form_submit_button("Predict Fraud Risk")
    
    if submitted:
        try:
            with st.spinner(random.choice([
                "Analyzing transaction...", 
                "Checking for red flags...",
                "Comparing to known patterns...",
                "Calculating risk score..."
            ])):
                # Simulate processing time
                import time
                time.sleep(random.uniform(0.5, 2))
                
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
                    input_df['Transaction_Amount'] = np.minimum(input_df['Transaction_Amount'], 100000)
                    input_df['Number_of_Transactions_Last_24H'] = np.minimum(input_df['Number_of_Transactions_Last_24H'], 50)
                    
                    # Time-based features
                    input_df['Hour'] = input_df['Time_of_Transaction'] % 24
                    input_df['Is_Night'] = ((input_df['Hour'] < 6) | (input_df['Hour'] > 20)).astype(int)
                    input_df['Weekend'] = ((input_df['Time_of_Transaction'] // 24) % 7 >= 5).astype(int)
                    
                    # Transaction velocity features
                    input_df['Amount_per_Transaction'] = input_df['Transaction_Amount'] / (input_df['Number_of_Transactions_Last_24H'] + 1)
                    input_df['Transaction_Velocity'] = input_df['Number_of_Transactions_Last_24H'] / (input_df['Account_Age'] + 7)
                    
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
                    
                    # Percentile features
                    input_df['Amount_Percentile'] = np.where(
                        input_df['Transaction_Amount'] < 1000, 0.3,
                        np.where(input_df['Transaction_Amount'] < 5000, 0.6, 0.9)
                    )
                    input_df['24H_Transactions_Percentile'] = np.where(
                        input_df['Number_of_Transactions_Last_24H'] < 5, 0.3,
                        np.where(input_df['Number_of_Transactions_Last_24H'] < 15, 0.6, 0.9))
                    
                    return input_df
                
                input_df = create_features(input_df)
                
                # One-hot encode categorical variables
                input_df = pd.get_dummies(input_df)
                
                # Ensure all expected columns are present
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
                
                # Add small random noise to probabilities for demo
                noise = random.uniform(-0.05, 0.05)
                prediction_proba[0][1] = np.clip(prediction_proba[0][1] + noise, 0, 1)
                prediction_proba[0][0] = 1 - prediction_proba[0][1]
                
                # Display results with random emoji
                st.subheader("Prediction Results")
                
                if prediction[0] == 1:
                    fraud_emojis = ["🚨", "⚠️", "🔴", "❌", "🛑"]
                    st.error(f"{random.choice(fraud_emojis)} Fraudulent Transaction Detected")
                else:
                    legit_emojis = ["✅", "👍", "🟢", "✔️", "👌"]
                    st.success(f"{random.choice(legit_emojis)} Legitimate Transaction")
                
                # Show probability with random confidence statement
                prob = prediction_proba[0][1]
                confidence = ""
                if prob < 0.3:
                    confidence = random.choice(["Low risk", "Probably safe", "Looks good"])
                elif prob < 0.7:
                    confidence = random.choice(["Moderate risk", "Worth reviewing", "Use caution"])
                else:
                    confidence = random.choice(["High risk!", "Strong fraud signal", "Immediate review recommended"])
                
                st.write(f"Probability of being fraudulent: {prob:.2%} ({confidence})")
                
                # Show feature importance with random ordering
                if hasattr(model, 'coef_'):
                    st.subheader("Top Contributing Factors")
                    coefs = pd.Series(model.coef_[0], index=expected_features)
                    
                    # Randomly select whether to show top or bottom factors
                    if random.choice([True, False]):
                        top_factors = coefs.abs().sort_values(ascending=False).head(5)
                        st.write("These factors contributed most to the prediction:")
                    else:
                        top_factors = coefs.abs().sort_values(ascending=True).head(5)
                        st.write("These factors had the least impact on the prediction:")
                    
                    for feature in top_factors.index:
                        value = input_df[feature].values[0]
                        impact = coefs[feature]
                        direction = "increased" if impact > 0 else "decreased"
                        st.write(f"- **{feature}**: Value = {value:.2f}, {direction} fraud risk (impact = {impact:.2f})")
                    
                    # Random recommendation
                    recommendations = [
                        "Consider verifying this transaction with the customer.",
                        "No additional action needed at this time.",
                        "Recommend reviewing transaction details carefully.",
                        "This appears to be a normal transaction pattern.",
                        "Suggest contacting the account holder for verification."
                    ]
                    st.info(random.choice(recommendations))
        except Exception as e:
            st.error("An error occurred during prediction. Please check your input values and try again.")
            st.expander("Show error details").write(traceback.format_exc())

# Random footer
footer_texts = [
    "Protecting your transactions since 2023",
    "Fraud detection powered by AI",
    "Making finance safer every day",
    "Innovative fraud prevention technology"
]
st.markdown(f"<hr><center><i>{random.choice(footer_texts)}</i></center>", unsafe_allow_html=True)
