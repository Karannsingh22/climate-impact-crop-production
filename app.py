import os

import gdown
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Climate Impact on Crop Production",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .info-card {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
        }

        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }

        .small-text {
            font-size: 14px;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

os.makedirs("models", exist_ok=True)

FILES = {
    "yield_prediction_model_RFR.pkl": "15Wdg_B2nfIyFld--CMwOBNlfPjJB-cLK",
    "crop_recommendation_RFC.pkl": "1Q7VfYU9OHgRbdj4MmFdiM4IT_0N85Yen",
    "label_encoder.pkl": "1EvlGD49RWYM91tJyE-__LC3ho6OOq-TV",
    "yield_columns.pkl": "1-DDWqQzepnQIoFZrrwMbPcjvlH7FfBmV",
}


@st.cache_resource
def download_and_load_models():

    for filename, file_id in FILES.items():

        path = os.path.join("models", filename)

        if not os.path.exists(path):

            url = f"https://drive.google.com/uc?id={file_id}"

            with st.spinner(f"Downloading {filename}..."):
                gdown.download(
                    url,
                    path,
                    quiet=False
                )

    yield_model = joblib.load(
        "models/yield_prediction_model_RFR.pkl"
    )

    crop_model = joblib.load(
        "models/crop_recommendation_RFC.pkl"
    )

    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    yield_columns = joblib.load(
        "models/yield_columns.pkl"
    )

    return (
        yield_model,
        crop_model,
        label_encoder,
        yield_columns,
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        yield_model,
        crop_model,
        label_encoder,
        yield_columns,
    ) = download_and_load_models()

except Exception as e:

    st.error(
        "Unable to load the machine learning models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# LOOKUP TABLES
# ============================================================

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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Climate Impact on Crop Production</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Machine Learning powered platform for crop yield prediction
    and intelligent crop recommendation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

with st.expander("📖 About This Project", expanded=True):

    st.markdown(
        """
        ### 🌾 What does this application do?

        This application uses Machine Learning to analyze agricultural
        and environmental conditions and provide useful predictions
        for crop production.

        The project contains two major machine learning applications:

        **1. Crop Yield Prediction**

        Predicts the expected crop yield using historical agricultural
        and environmental information such as:

        - Country
        - Crop
        - Year
        - Rainfall
        - Pesticide usage
        - Average temperature

        **2. Crop Recommendation**

        Recommends a suitable crop based on soil and environmental
        conditions such as:

        - Nitrogen (N)
        - Phosphorus (P)
        - Potassium (K)
        - Temperature
        - Humidity
        - Soil pH
        - Rainfall

        The objective is to demonstrate how Machine Learning can be
        applied to real-world agricultural and climate-related problems.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 Project Information")

    st.markdown(
        """
        ### Machine Learning Tasks

        **🌾 Regression**
        - Crop Yield Prediction
        - Random Forest Regressor

        **🌱 Classification**
        - Crop Recommendation
        - Random Forest Classifier

        **🔬 Clustering**
        - K-Means Clustering
        """
    )

    st.divider()

    st.markdown(
        """
        ### 🛠️ Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit
        - Google Drive
        """
    )

    st.divider()

    st.caption(
        "Built as an end-to-end Machine Learning project."
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🌾 Yield Prediction",
        "🌱 Crop Recommendation",
        "📊 Model & Project Information",
    ]
)


# ============================================================
# TAB 1 — YIELD PREDICTION
# ============================================================

with tab1:

    st.header("🌾 Crop Yield Prediction")

    st.markdown(
        """
        Predict the expected crop yield based on historical
        agricultural and climate-related conditions.
        """
    )

    st.info(
        "Enter the country and crop exactly as they appear in the dataset. "
        "The model uses historical agricultural and environmental patterns "
        "to estimate yield."
    )

    col1, col2 = st.columns(2)

    with col1:

        area = st.text_input(
            "🌍 Country",
            placeholder="Example: India",
            help="Country/area used in the historical dataset.",
        ).strip().lower()

        item = st.text_input(
            "🌾 Crop",
            placeholder="Example: Wheat",
            help="Crop for which you want to estimate yield.",
        ).strip().lower()

        year = st.number_input(
            "📅 Year",
            min_value=1900,
            max_value=2100,
            value=2020,
            step=1,
            help="Year for which the prediction is being made.",
        )

    with col2:

        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            value=1000.0,
            step=10.0,
            help="Average annual rainfall in millimetres.",
        )

        pesticides = st.number_input(
            "🧪 Pesticides (tonnes)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Amount of pesticide usage associated with the crop/area.",
        )

        temp = st.number_input(
            "🌡️ Average Temperature (°C)",
            value=20.0,
            step=0.1,
            help="Average temperature in degrees Celsius.",
        )

    st.divider()

    if st.button(
        "🔮 Predict Crop Yield",
        type="primary",
        use_container_width=True,
    ):

        if not area:

            st.warning("Please enter a country.")

        elif not item:

            st.warning("Please enter a crop.")

        else:

            area_col = area_lookup.get(area)
            item_col = item_lookup.get(item)

            if not area_col:

                st.error(
                    f"Country '{area.title()}' was not found in the dataset."
                )

                st.info(
                    "Try checking the country name or use a country "
                    "available in the training dataset."
                )

            elif not item_col:

                st.error(
                    f"Crop '{item.title()}' was not found in the dataset."
                )

                st.info(
                    "Try checking the crop name or use a crop available "
                    "in the training dataset."
                )

            else:

                try:

                    input_df = pd.DataFrame(
                        0,
                        index=[0],
                        columns=yield_columns,
                    )

                    input_df["Year"] = year

                    input_df[
                        "average_rain_fall_mm_per_year"
                    ] = rainfall

                    input_df["pesticides_tonnes"] = pesticides

                    input_df["avg_temp"] = temp

                    input_df[area_col] = 1

                    input_df[item_col] = 1

                    prediction = yield_model.predict(
                        input_df
                    )

                    predicted_yield = prediction[0]

                    st.success(
                        "Prediction completed successfully!"
                    )

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:

                        st.metric(
                            "Predicted Crop Yield",
                            f"{predicted_yield:,.2f}",
                        )

                    with metric_col2:

                        st.metric(
                            "Unit",
                            "hg/ha",
                        )

                    st.markdown(
                        f"""
                        ### 🌾 Prediction Result

                        Based on the provided environmental and
                        agricultural conditions, the estimated yield
                        for **{item.title()}** in **{area.title()}**
                        is:

                        ## {predicted_yield:,.2f} hg/ha
                        """
                    )

                except Exception as e:

                    st.error(
                        "An error occurred while generating the prediction."
                    )

                    st.exception(e)


# ============================================================
# TAB 2 — CROP RECOMMENDATION
# ============================================================

with tab2:

    st.header("🌱 Crop Recommendation System")

    st.markdown(
        """
        Enter the soil and environmental conditions of your agricultural
        land. The classification model will recommend the crop that best
        matches these conditions.
        """
    )

    st.info(
        "The recommendation is based on the relationship between soil "
        "nutrients and environmental conditions learned from the training data."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🧪 Soil Conditions")

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=50.0,
            help="Nitrogen content in the soil.",
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=50.0,
            help="Phosphorus content in the soil.",
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=50.0,
            help="Potassium content in the soil.",
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            help="Acidity or alkalinity of the soil.",
        )

    with col2:

        st.subheader("🌦️ Climate Conditions")

        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.1,
            help="Average temperature.",
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
            help="Relative humidity.",
        )

        rainfall2 = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Expected or average rainfall.",
        )

    st.divider()

    if st.button(
        "🌱 Recommend Crop",
        type="primary",
        use_container_width=True,
    ):

        try:

            input_data = pd.DataFrame(
                {
                    "N": [N],
                    "P": [P],
                    "K": [K],
                    "temperature": [temperature],
                    "humidity": [humidity],
                    "ph": [ph],
                    "rainfall": [rainfall2],
                }
            )

            pred = crop_model.predict(
                input_data
            )

            crop = label_encoder.inverse_transform(
                pred
            )[0]

            st.success(
                "Crop recommendation generated successfully!"
            )

            st.markdown(
                f"""
                ### 🌾 Recommended Crop

                # {crop.title()}

                Based on the soil and environmental conditions provided,
                **{crop.title()}** is the crop recommended by the
                machine learning model.
                """
            )

            st.info(
                "This recommendation is generated by a machine learning "
                "model and should be treated as a decision-support tool."
            )

        except Exception as e:

            st.error(
                "An error occurred while generating the recommendation."
            )

            st.exception(e)


# ============================================================
# TAB 3 — MODEL & PROJECT INFORMATION
# ============================================================

with tab3:

    st.header("📊 Model & Project Information")

    st.subheader("🤖 Machine Learning Models")

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown(
            """
            ### 🌾 Crop Yield Prediction

            **Algorithm:** Random Forest Regressor

            **Problem Type:** Regression

            **Target:** Crop Yield

            **Evaluation Metrics:**

            - R² Score
            - RMSE
            - MAE

            The regression model learns relationships between historical
            crop production, climate conditions, pesticide usage and
            agricultural regions to estimate crop yield.
            """
        )

    with model_col2:

        st.markdown(
            """
            ### 🌱 Crop Recommendation

            **Algorithm:** Random Forest Classifier

            **Problem Type:** Classification

            **Target:** Crop Type

            **Input Features:**

            - Nitrogen
            - Phosphorus
            - Potassium
            - Temperature
            - Humidity
            - Soil pH
            - Rainfall

            The classifier predicts the crop class that best matches
            the provided soil and environmental conditions.
            """
        )

    st.divider()

    st.subheader("🔬 Clustering")

    st.markdown(
        """
        The project also includes **K-Means Clustering** to identify
        groups of similar agricultural conditions.

        The **Elbow Method** was used during analysis to help determine
        an appropriate number of clusters.

        Clustering is an unsupervised learning technique, meaning that
        the algorithm identifies patterns without predefined target labels.
        """
    )

    st.divider()

    st.subheader("📚 Dataset Features")

    feature_data = pd.DataFrame(
        {
            "Feature": [
                "N",
                "P",
                "K",
                "Temperature",
                "Humidity",
                "pH",
                "Rainfall",
                "Pesticides",
                "Year",
                "Country",
                "Crop",
            ],
            "Description": [
                "Nitrogen content in soil",
                "Phosphorus content in soil",
                "Potassium content in soil",
                "Average temperature",
                "Relative humidity",
                "Soil acidity/alkalinity",
                "Rainfall received",
                "Pesticide usage",
                "Agricultural year",
                "Country/area",
                "Crop type",
            ],
        }
    )

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.markdown(
            """
            **Programming & Data**

            - Python
            - Pandas
            - NumPy
            """
        )

    with tech_col2:

        st.markdown(
            """
            **Machine Learning**

            - Scikit-learn
            - Random Forest
            - K-Means
            - Joblib
            """
        )

    with tech_col3:

        st.markdown(
            """
            **Deployment**

            - Streamlit
            - GitHub
            - Google Drive
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 Climate Impact on Crop Production | "
    "Machine Learning & Data Analytics Project"
)

