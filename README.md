# 🌱 Climate Impact on Crop Production

An end-to-end Machine Learning project that analyzes the impact of climate and environmental factors on agriculture and provides **crop yield prediction, crop recommendation, and crop clustering**.

The project includes exploratory data analysis, machine learning model development, evaluation, and an interactive **Streamlit web application** for making predictions.

## 🚀 Live Demo

**Streamlit App:**
*Add your deployed Streamlit URL here after deployment.*

---

## 📌 Project Overview

Agricultural production is strongly influenced by environmental and climatic conditions such as:

* Temperature
* Rainfall
* Humidity
* Soil nutrients
* Soil pH
* Crop type
* Geographic and seasonal conditions

This project uses historical agricultural and environmental data to build machine learning solutions that can assist with crop planning and yield estimation.

### The project focuses on three major tasks:

1. **Crop Yield Prediction** – Predict the expected crop yield based on agricultural and environmental factors.
2. **Crop Recommendation** – Recommend a suitable crop based on soil and climatic conditions.
3. **Crop Clustering** – Group similar crops/conditions using unsupervised learning.

---

## 🎯 Objectives

* Analyze agricultural and climate-related datasets.
* Perform data cleaning and exploratory data analysis.
* Identify relationships between environmental factors and crop production.
* Build machine learning models for crop yield prediction.
* Build a crop recommendation system.
* Apply clustering to identify similar agricultural patterns.
* Evaluate model performance using appropriate metrics.
* Deploy the final application using Streamlit.

---

## 📊 Dataset

The project uses multiple agricultural and environmental datasets containing information related to crop production, weather, soil conditions, and agricultural inputs.

### Major Features

| Feature     | Description                |
| ----------- | -------------------------- |
| N           | Nitrogen content in soil   |
| P           | Phosphorus content in soil |
| K           | Potassium content in soil  |
| Temperature | Average temperature        |
| Humidity    | Relative humidity          |
| pH          | Soil pH value              |
| Rainfall    | Rainfall received          |
| Crop        | Type of crop               |
| Area        | Agricultural area          |
| Production  | Total crop production      |
| Yield       | Crop yield                 |

Additional datasets contain information about:

* Rainfall
* Temperature
* Pesticide usage
* Crop production
* Crop yield

---

## 🔍 Exploratory Data Analysis

The project performs Exploratory Data Analysis (EDA) to understand:

* Dataset distributions
* Missing values
* Outliers
* Correlations between variables
* Climate and crop relationships
* Crop-wise production patterns
* Yield trends

Various visualizations are used to identify important patterns and relationships within the data.

---

# 🤖 Machine Learning

## 1. 🌾 Crop Yield Prediction

A **Random Forest Regressor** is used to predict crop yield based on relevant agricultural and environmental features.

### Problem Type

**Regression**

### Model

`RandomForestRegressor`

### Evaluation Metrics

* R² Score
* RMSE
* MAE

---

## 2. 🌱 Crop Recommendation

A **Random Forest Classifier** is used to recommend the most suitable crop based on soil and environmental conditions.

### Input Features

* Nitrogen
* Phosphorus
* Potassium
* Temperature
* Humidity
* pH
* Rainfall

### Problem Type

**Classification**

### Model

`RandomForestClassifier`

---

## 3. 🔬 Crop Clustering

**K-Means Clustering** is used to identify groups of similar agricultural conditions/crops.

The **Elbow Method** is used to help determine an appropriate number of clusters.

### Technique

`KMeans`

### Purpose

Clustering helps identify similarities and patterns within agricultural data without requiring predefined labels.

---

# 📈 Model Performance

The application provides model performance information and visualizations to help understand how well the machine learning models perform.

### Metrics Used

**Regression**

* R² Score
* RMSE
* MAE

**Classification**

* Accuracy
* Precision
* Recall
* F1-Score

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit application where users can:

### 🌾 Predict Crop Yield

Enter relevant agricultural and environmental parameters to estimate crop yield.

### 🌱 Recommend a Crop

Enter:

```text
Nitrogen
Phosphorus
Potassium
Temperature
Humidity
pH
Rainfall
```

and receive a recommended crop.

### 📊 Explore Data

The application provides visualizations and information about the datasets and model results.

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* K-Means Clustering

### Model Management

* Joblib

### Web Application

* Streamlit

### Deployment

* Streamlit Community Cloud
* GitHub

---

# 📁 Project Structure

```text
Cimate_Impact_on_Crop_Production/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── Crop_recommendation.csv
│   ├── pesticides.csv
│   ├── rainfall.csv
│   ├── temp.csv
│   ├── yield.csv
│   ├── yield_df.csv
│   └── yield_df_cleaned.csv
│
└── notebooks/
    ├── EDA.ipynb
    ├── CropYieldPrediction.ipynb
    ├── CropRecommendationSystem.ipynb
    └── Clustering.ipynb
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/climate-impact-crop-production.git
```

### 2. Navigate to the project

```bash
cd climate-impact-crop-production
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Select the `main` branch.
6. Set the main file as:

```text
app.py
```

7. Click **Deploy**.

---

# 🔮 Future Improvements

Possible future improvements include:

* Integration of real-time weather data.
* Location-based crop recommendations.
* More advanced ensemble models.
* XGBoost/LightGBM comparison.
* Hyperparameter optimization.
* Explainable AI using SHAP.
* Improved model monitoring.
* Real-time agricultural dashboards.
* Integration with satellite or remote-sensing data.

---

# 👨‍💻 Author

**Karan Singh**

B.Tech Student | Machine Learning & Data Analytics

---

## ⭐ Project Highlights

* End-to-end Machine Learning project
* Regression + Classification + Clustering
* Exploratory Data Analysis
* Multiple agricultural datasets
* Interactive Streamlit application
* GitHub-based deployment
* Real-world agriculture and climate use case
