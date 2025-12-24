# 🏡 Melbourne House Price Prediction

An end-to-end Machine Learning regression project that predicts house prices in Melbourne using Decision Tree and Random Forest models.

This project focuses on the complete ML workflow — from data preprocessing to model evaluation and tuning.

---

## 📌 Problem Statement
Predict the **price of houses in Melbourne** based on various features such as:
- Number of rooms
- Bathrooms
- Land size
- Building area
- Year built
- Latitude & Longitude

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 📊 Dataset
- **Source:** Melbourne Housing Dataset  
- Contains housing attributes and sale prices.
- Missing values handled using row-wise removal.

---

## 🔍 Features Used
- Rooms  
- Bathroom  
- Landsize  
- BuildingArea  
- YearBuilt  
- Lattitude  
- Longtitude  

Target Variable:
- **Price**

---

## 🤖 Models Implemented

### 1️⃣ Decision Tree Regressor
- Baseline model
- Evaluated using Mean Absolute Error (MAE)

### 2️⃣ Model Tuning
- Controlled overfitting using `max_leaf_nodes`
- Compared MAE across different tree sizes

### 3️⃣ Random Forest Regressor
- Ensemble model for improved accuracy
- Achieved lower MAE compared to Decision Tree

---

## 📈 Evaluation Metric
- **Mean Absolute Error (MAE)**

---

## 🚀 Results
- Random Forest outperformed Decision Tree
- Model tuning significantly reduced overfitting
- Demonstrates importance of validation and model comparison

---


