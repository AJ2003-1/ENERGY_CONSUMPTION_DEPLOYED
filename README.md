# ⚡ Household Energy Consumption Prediction
Machine Learning Regression Project using XGBoost

---

## 📌 Overview

Accurate forecasting of household electricity consumption is critical for energy optimization, demand planning, and smart grid systems.  
This project focuses on predicting household **Global Active Power (kW)** using **XGBoost regression**, after comparing multiple machine learning models.  

The final model achieved **R² = 0.9988** and extremely low error rates on test data, making it highly reliable for real-world deployment.

---

## 🎯 Problem Statement

Electricity consumption patterns are influenced by time, human behavior, and appliance usage, making them highly non-linear and difficult to model using traditional approaches.

**Objective:**
- Predict household energy consumption in **kW**
- Evaluate multiple regression models and select the best-performing model
- Deploy a production-ready model for real-time predictions

---

## 📊 Dataset Information

- **Dataset:** Household Power Consumption Dataset  
- **Source:** Kaggle  
- **Original Provider:** UCI Machine Learning Repository  
- **Frequency:** 1-minute intervals  
- **Time Span:** Multi-year household energy usage  

### Key Features
- `Global_active_power` *(Target – kW)*
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`
- `Date`, `Time`

---

## 🧠 Methodology

### Data Preprocessing
- Handled missing values
- Combined `Date` and `Time` into a single datetime feature
- Removed redundant columns after feature extraction

### Feature Engineering
Time-based features were extracted to capture usage patterns:
- Hour
- Day
- Month
- Day of week
- Weekend indicator

### Target Transformation
- Applied **log transformation (`log1p`)** to reduce skewness and improve model stability

---

## 🤖 Model Selection

Multiple regression models were trained and evaluated:

| Model             | RMSE      | MAE       | R² Score  |
|------------------|-----------|-----------|-----------|
| XGBoost          | 0.014777  | 0.009963  | 0.998814  |
| Decision Tree     | 0.014819  | 0.006829  | 0.998808  |
| Linear Regression | 0.086352  | 0.060147  | 0.959514  |

After evaluation, **XGBoost was selected as the best model** due to its superior predictive performance and ability to capture complex non-linear patterns in energy consumption.

---

## 📏 Model Performance

**Best Model:** XGBoost Regressor  
**Metrics:**
- **RMSE:** 0.0148  
- **MAE:** 0.00996  
- **R² Score:** 0.9988  

---

## 🏆 Model Saving & Deployment

The trained XGBoost model is saved for deployment:

### Saved Artifact
