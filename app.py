import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# --- Custom Styling (Fixed unsafe_allow_html) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Data & Models ---
@st.cache_resource
def load_assets():
    # Ensure these files are in the same directory as your app.py
    data = pd.read_csv('train.csv')
    model = pickle.load(open('randomforest.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    return data, model, preprocessor

try:
    data, model, preprocessor = load_assets()
except FileNotFoundError:
    st.error("Error: Model or Data files not found. Please ensure 'train.csv', 'randomforest.pkl', and 'preprocessor.pkl' are in the project folder.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609036.png", width=100)
    st.title("Developer Gallery")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed by:")
    st.write("**Shubham Kumar**")
    st.caption("M.Tech AI & Machine Learning")
    
    st.markdown("---")
    st.markdown("### 🛠️ Model Info")
    st.info("Algorithm: Random Forest Regressor\n\nThis ensemble model averages multiple decision trees to provide a robust price estimation.")
    
    st.markdown("[🔗 GitHub Profile](https://github.com/shubham7667)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/shubham~kumar/)")

# --- Main UI ---
st.title("🏠 House Price Predictor")
st.markdown("Fill in the details below to estimate the market value of the property.")
st.write("---")

# Using Columns for a spacious layout
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.subheader("📍 Property Context")
    neighborhood = st.selectbox("Neighborhood", sorted(data['Neighborhood'].dropna().unique()))
    house_age = st.number_input("House Age (Years)", min_value=0, max_value=150, value=10, step=1)
    
    st.subheader("⭐ Quality & Condition")
    overall_qual = st.slider("Material Quality (1-10)", 1, 10, 5)
    overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)

with col2:
    st.subheader("📐 Living Space")
    indoor_area = st.number_input("Indoor Area (sq ft)", min_value=0.0, value=1500.0, step=50.0)
    outdoor_area = st.number_input("Outdoor Area (sq ft)", min_value=0.0, value=500.0, step=50.0)
    
    st.subheader("🚗 Garage Details")
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0.0, value=400.0)
    garage_cars = st.number_input("Garage Car Capacity", min_value=0, max_value=5, value=2)
    
with col3:
    st.subheader("🛌 Rooms & Baths")
    bedrooms = st.number_input("Total Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Total Bathrooms", min_value=0, max_value=10, value=2)
    
    st.info("💡 Tip: Total area is calculated automatically as the sum of indoor and outdoor space.")

st.write("---")

# --- Prediction Logic ---
if st.button("Generate Price Estimate"):
    with st.spinner('Calculating valuation...'):
        # 1. Create a template DataFrame from the training data structure
        input_df = data.drop('SalePrice', axis=1).iloc[0:1].copy()

        # 2. Update the template with User Inputs
        input_df['OverallQual'] = overall_qual
        input_df['OverallCond'] = overall_cond
        input_df['BedroomAbvGr'] = bedrooms
        input_df['GarageArea'] = garage_area
        input_df['GarageCars'] = garage_cars
        input_df['Neighborhood'] = neighborhood

        # Engineered features matching your training pipeline
        input_df['indoor_area'] = indoor_area
        input_df['outdoor_area'] = outdoor_area
        input_df['total_area'] = indoor_area + outdoor_area
        input_df['total_bathroom'] = bathrooms
        input_df['house_age'] = house_age
        input_df['renovation_age'] = house_age  # Matches your simplified assumption

        # 3. Transform and Predict
        try:
            processed_input = preprocessor.transform(input_df)
            prediction = model.predict(processed_input)

            # 4. Display Result
            st.balloons()
            st.markdown(f"""
                <div style="text-align: center; padding: 30px; border-radius: 15px; background-color: #e3f2fd; border: 2px solid #2196f3; margin-top: 20px;">
                    <h3 style="color: #1565c0; margin-bottom: 10px;">The Estimated Market Value is:</h3>
                    <h1 style="color: #0d47a1; font-size: 55px; margin: 0;">${prediction[0]:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("<br><br><center><small>Model trained on the Ames Housing Dataset</small></center>", unsafe_allow_html=True)
