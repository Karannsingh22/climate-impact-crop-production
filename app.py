import streamlit as st
import pandas as pd
import joblib
import gdown
import os

os.makedirs("models", exist_ok=True)

FILES = {
    "yield_prediction_model_RFR.pkl": "15Wdg_B2nfIyFld--CMwOBNlfPjJB-cLK",
    "crop_recommendation_RFC.pkl": "1Q7VfYU9OHgRbdj4MmFdiM4IT_0N85Yen",
    "label_encoder.pkl": "1EvlGD49RWYM91tJyE-__LC3ho6OOq-TV",
    "yield_columns.pkl": "1-DDWqQzepnQIoFZrrwMbPcjvlH7FfBmV",
}

for filename, file_id in FILES.items():
    path = f"models/{filename}"

    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

yield_model = joblib.load("models/yield_prediction_model_RFR.pkl")
crop_model = joblib.load("models/crop_recommendation_RFC.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
yield_columns = joblib.load("models/yield_columns.pkl")

area_lookup = {
    col.replace("Area_", "").lower(): col
    for col in yield_columns
    if col.startswith("Area_")
}

item_lookup = {
    col.replace("Item_", "").lower(): col
    for col in yield_columns
    if col.startswith("Item_")
}


st.set_page_config(
    page_title="Agriculture ML Project",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Climate Impact on Crop Production")

tab1, tab2 = st.tabs(["Yield Prediction", "Crop Recommendation"])


with tab1:

    st.header("Crop Yield Prediction")

    area = st.text_input("Country").strip().lower()
    item = st.text_input("Crop").strip().lower()

    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2100,
        value=2020
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0
    )

    pesticides = st.number_input(
        "Pesticides (tonnes)",
        min_value=0.0
    )

    temp = st.number_input(
        "Average Temperature (°C)"
    )

    if st.button("Predict Yield"):

        input_df = pd.DataFrame(
            0,
            index=[0],
            columns=yield_columns
        )

        input_df['Year'] = year
        input_df['average_rain_fall_mm_per_year'] = rainfall
        input_df['pesticides_tonnes'] = pesticides
        input_df['avg_temp'] = temp

        # Match country
        area_col = area_lookup.get(area)

        if area_col:
            input_df[area_col] = 1
        else:
            st.warning("Country not found in dataset.")

        # Match crop
        item_col = item_lookup.get(item)

        if item_col:
            input_df[item_col] = 1
        else:
            st.warning("Crop not found in dataset.")

        prediction = yield_model.predict(input_df)

        st.success(
            f"Predicted Yield = {prediction[0]:.2f} hg/ha"
        )


with tab2:

    st.header("Crop Recommendation")

    N = st.number_input("Nitrogen (N)", min_value=0.0)

    P = st.number_input("Phosphorus (P)", min_value=0.0)

    K = st.number_input("Potassium (K)", min_value=0.0)

    temperature = st.number_input("Temperature")

    humidity = st.number_input(
        "Humidity",
        min_value=0.0,
        max_value=100.0
    )

    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0
    )

    rainfall2 = st.number_input(
        "Rainfall",
        min_value=0.0
    )

    if st.button("Recommend Crop"):

        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall2]
        })

        pred = crop_model.predict(input_data)

        crop = label_encoder.inverse_transform(pred)[0]

        st.success(
            f"Recommended Crop: {crop}"
        )