import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Title and description
st.title("💻 Laptop Price Predictor")
st.markdown("Predict laptop prices based on specifications")

# Function to load and preprocess data
@st.cache_resource
def load_data():
    # Load dataset (replace 'laptops.csv' with your dataset path)
    df = pd.read_csv('laptops.csv')
    
    # Preprocessing
    df = df.dropna()
    df = df[df['Price'] > 0]  # Remove invalid prices
    
    # Encode categorical features
    le_brand = LabelEncoder()
    le_processor = LabelEncoder()
    
    df['Brand_encoded'] = le_brand.fit_transform(df['Brand'])
    df['Processor_encoded'] = le_processor.fit_transform(df['Processor'])
    
    return df, le_brand, le_processor

# Function to train and save model
def train_model(df):
    X = df[['Brand_encoded', 'Processor_encoded', 'RAM_GB', 'Storage_GB', 'Screen_Size']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'laptop_model.pkl')
    return model

# Load data and model
try:
    df, le_brand, le_processor = load_data()
    model = joblib.load('laptop_model.pkl')
except:
    st.warning("Training model for the first time...")
    df, le_brand, le_processor = load_data()
    model = train_model(df)

# Input sidebar
st.sidebar.header("🛠️ Laptop Specifications")
brand = st.sidebar.selectbox("Brand", options=le_brand.classes_)
processor = st.sidebar.selectbox("Processor", options=le_processor.classes_)
ram = st.sidebar.slider("RAM (GB)", 4, 64, 8)
storage = st.sidebar.slider("Storage (GB)", 128, 4096, 512)
screen_size = st.sidebar.slider("Screen Size (inches)", 10.0, 17.3, 15.6)

# Prediction function
def predict_price():
    # Encode inputs
    brand_encoded = le_brand.transform([brand])[0]
    processor_encoded = le_processor.transform([processor])[0]
    
    # Create input array
    input_data = [[brand_encoded, processor_encoded, ram, storage, screen_size]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Calculate range (using standard deviation from training data)
    std_dev = np.std([tree.predict(input_data) for tree in model.estimators_], axis=0)[0]
    lower_bound = max(0, prediction - 1.5 * std_dev)
    upper_bound = prediction + 1.5 * std_dev
    
    return prediction, lower_bound, upper_bound

# Prediction button
if st.sidebar.button("Predict Price", type="primary"):
    price, low, high = predict_price()
    
    # Display results
    st.success(f"### Predicted Price: ${price:,.2f}")
    st.info(f"**Price Range:** ${low:,.2f} - ${high:,.2f}")
    
    # Visual indicator
    st.metric("Price Estimate", f"${price:,.2f}", delta=f"±${(high-low)/2:,.0f} range")
    
    # Price distribution visualization
    chart_data = pd.DataFrame({
        'Estimate': [price],
        'Lower Bound': [low],
        'Upper Bound': [high]
    })
    st.area_chart(chart_data.T, use_container_width=True)

# Show sample data
if st.checkbox("Show sample data"):
    st.dataframe(df[['Brand', 'Processor', 'RAM_GB', 'Storage_GB', 'Screen_Size', 'Price']].head(10))

# How to use
st.expander("ℹ️ How to use this app").markdown("""
1. Select laptop specifications using the controls in the sidebar
2. Click the 'Predict Price' button
3. View the predicted price and estimated range
4. Explore sample data using the checkbox
""")
    