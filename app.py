import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("linear_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Taxi Trip Price Predictor")

# User Inputs
trip_distance = st.number_input("Trip Distance (km)", min_value=0.0)
passenger_count = st.number_input("Passenger Count", min_value=1, step=1)
base_fare = st.number_input("Base Fare")
per_km_rate = st.number_input("Per Km Rate")
per_min_rate = st.number_input("Per Minute Rate")
trip_duration = st.number_input("Trip Duration (Minutes)")

time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])
traffic = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])

if st.button("Predict Price"):

    input_dict = {
        'Trip_Distance_km': trip_distance,
        'Passenger_Count': passenger_count,
        'Base_Fare': base_fare,
        'Per_Km_Rate': per_km_rate,
        'Per_Minute_Rate': per_min_rate,
        'Trip_Duration_Minutes': trip_duration,
        f'Time_of_Day_{time_of_day}': 1,
        f'Day_of_Week_{day_of_week}': 1,
        f'Traffic_Conditions_{traffic}': 1,
        f'Weather_{weather}': 1
    }

    input_df = pd.DataFrame([input_dict])

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Estimated Trip Price: ${prediction[0]:.2f}")
