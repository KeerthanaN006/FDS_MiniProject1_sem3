import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data cleaning functions
def convert_price(value):
    value = str(value).replace('₹', '').replace(',', '').strip()
    
    # Handle price ranges (e.g., '5.2 - 6.3')
    if ' - ' in value:
        low, high = value.split(' - ')
        try:
            low = float(low.strip())
            high = float(high.strip())
            return (low + high) / 2 * 10**7 
        except ValueError:
            return np.nan
    
    # Convert from 'Cr' or 'L'
    if 'Cr' in value:
        try:
            return float(value.replace(' Cr', '').strip()) * 10**7
        except ValueError:
            return np.nan
    elif 'L' in value:
        try:
            return float(value.replace(' L', '').strip()) * 10**5
        except ValueError:
            return np.nan
    else:
        try:
            return float(value)
        except ValueError:
            return np.nan

def convert_sqft(value):
    if isinstance(value, str):
        try:
            return float(value.replace('sq.ft.', '').replace(',', '').strip())
        except ValueError:
            return np.nan
    elif isinstance(value, (float, int)):
        return value
    else:
        return np.nan

# Load dataset and apply cleaning
df = pd.read_csv('gurgaon_10k.csv')
df['PRICE'] = df['PRICE'].apply(convert_price)
df['BUILTUP_SQFT'] = df['BUILTUP_SQFT'].apply(convert_sqft)

# Drop rows with missing or invalid values in PRICE or BUILTUP_SQFT
df = df.dropna(subset=['PRICE', 'BUILTUP_SQFT'])

# Streamlit app setup
st.set_page_config(page_title="GharConnect", layout="wide")

# logo and custom sidebar
st.sidebar.image("logo.png", width=120)
st.sidebar.title("Filters")
bedroom_num = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=5, value=3)
bathroom_num = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, value=2)
min_price = st.sidebar.number_input("Minimum Price (in Lakhs)", min_value=0, value=50)
max_price = st.sidebar.number_input("Maximum Price (in Lakhs)", min_value=100, value=500)

# Page title and description
st.markdown("<h1 style='text-align: center; color: white;'>GharConnect</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Apna Ghar Humara Vaada</p>", unsafe_allow_html=True)

# Apply user filters
df_filtered = df[(df['BEDROOM_NUM'] == bedroom_num) & 
                 (df['BATHROOM_NUM'] == bathroom_num) & 
                 (df['PRICE'] >= min_price * 1e5) & 
                 (df['PRICE'] <= max_price * 1e5)]

# If there are matching properties, display them
if not df_filtered.empty:
    st.write(f"Found {len(df_filtered)} matching properties!")
    st.dataframe(df_filtered[['LOCALITY', 'PRICE', 'BUILTUP_SQFT', 'BEDROOM_NUM', 'BATHROOM_NUM']])
else:
    st.write("No matching properties found!")

# Linear regression model for price prediction
X = df[['BUILTUP_SQFT', 'BEDROOM_NUM', 'BATHROOM_NUM']]
y = df['PRICE']

# Drop rows with NaN values in X
X = X.dropna()

# Make sure that X and y have the same length after dropping NaN values
y = y.loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Plot the predicted vs actual prices in a square frame
fig, ax = plt.subplots(figsize=(5,5))  # Make the plot square
sns.regplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Prices")
st.pyplot(fig)

# New slide for additional graphs
if st.button("Show More Graphs"):
    # Add your additional graphs here
    fig, ax = plt.subplots(figsize=(5,5))  
    sns.histplot(df['PRICE'], bins=30, ax=ax)
    ax.set_title('Distribution of Property Prices')
    st.pyplot(fig)

# Predict property price option
st.sidebar.title("Predict Property Price")
builtup_sqft = st.sidebar.number_input("Built-up Area (in sq.ft.)", min_value=500, max_value=5000, value=1000)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=5, value=2)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=5, value=2)

if st.sidebar.button("Predict Price"):
    user_input = np.array([[builtup_sqft, bedrooms, bathrooms]])
    predicted_price = pipeline.predict(user_input)[0]
    st.sidebar.write(f"Predicted Price: ₹{predicted_price / 1e5:.2f} Lakhs")

# App layout customizations (background color and styles)
st.markdown("""
    <style>
    body {
        background-color: #87CEEB;  /* Sea blue */
    }
    .sidebar .sidebar-content {
        background-color: white;  /* White navbar */
    }
    .stButton>button {
        background-color: #f44336;  /* Button color */
        color: white;
    }
    .stSlider>div[role='slider'] {
        color: black;
    }
    img {
        border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)
