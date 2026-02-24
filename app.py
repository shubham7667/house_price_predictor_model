import streamlit as st
import pandas as pd
import pickle
data = pd.read_csv('train.csv')
model = pickle.load(open('randomforest.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

st.header("🏠 House Price Predictor")

overall_qual = st.slider("Overall Quality", 1, 10)
overall_cond = st.slider("Overall Condition", 1, 10)

indoor_area = st.number_input("Indoor Area", min_value=0.0)
outdoor_area = st.number_input("Outdoor Area", min_value=0.0)

bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)

garage_area = st.number_input("Garage Area", min_value=0.0)
garage_cars = st.number_input("Garage Cars", min_value=0)

house_age = st.number_input("House Age", min_value=0)

neighborhood = st.selectbox(
    "Neighborhood",
    sorted(data['Neighborhood'].dropna().unique())
)

if st.button("Predict Price"):

    # 🔥 Step 1: Take a template row
    input_df = data.drop('SalePrice', axis=1).iloc[0:1].copy()

    # 🔥 Step 2: Replace values with user input
    input_df['OverallQual'] = overall_qual
    input_df['OverallCond'] = overall_cond
    input_df['BedroomAbvGr'] = bedrooms
    input_df['GarageArea'] = garage_area
    input_df['GarageCars'] = garage_cars
    input_df['Neighborhood'] = neighborhood

    # 🔥 Your engineered features
    input_df['indoor_area'] = indoor_area
    input_df['outdoor_area'] = outdoor_area
    input_df['total_area'] = indoor_area + outdoor_area
    input_df['total_bathroom'] = bathrooms
    input_df['house_age'] = house_age

    # ⚠️ If you created renovation_age during training
    input_df['renovation_age'] = house_age  # simple assumption

    # 🔥 Step 3: Transform
    processed_input = preprocessor.transform(input_df)

    # 🔥 Step 4: Predict
    prediction = model.predict(processed_input)

    # 🔥 Step 5: Show result
    st.success(f"💰 Estimated Price: {prediction[0]:,.2f}")