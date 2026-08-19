# 🏠 House Price Prediction

A Flask web application that predicts house prices using a Multiple Linear Regression (MLR) model trained on housing data (bedrooms, bathrooms, square footage, location, and more).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Step-by-Step Setup](#step-by-step-setup)
- [Model Training](#model-training)
- [Running the App Locally](#running-the-app-locally)
- [Deployment](#deployment)
- [Live Demo](#live-demo)
- [Future Improvements](#future-improvements)

---

## 🔎 Overview

This project trains a **Multiple Linear Regression** model on the King County–style house sales dataset and serves predictions through a simple **Flask** web interface. Users enter property details (bedrooms, bathrooms, square footage, city, sale date, etc.) and receive a predicted sale price.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                       # Flask application (loads model, handles predictions)
├── development.py               # Model training script (data prep, train/test, saves MLR.pkl)
├── house_price_prediction.csv   # Training dataset
├── MLR.pkl                      # Trained Linear Regression model (pickled)
├── requirements.txt             # Python dependencies
├── procfile                     # Deployment process definition (gunicorn)
└── templates/
    └── index.html                # Web form UI (input + prediction display)
```

> **Note:** Make sure a `templates/index.html` file exists, since `app.py` renders it for both the form and the results page.

---

## 🛠 Tech Stack

- **Python 3**
- **Flask** – web framework
- **scikit-learn** – Linear Regression model
- **pandas / numpy** – data processing
- **gunicorn** – production WSGI server

---

## 📊 Dataset

The dataset (`house_price_prediction.csv`) contains the following columns:

| Column | Description |
|---|---|
| `date` | Sale date |
| `price` | Sale price (target variable) |
| `bedrooms`, `bathrooms` | Room counts |
| `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement` | Area measurements |
| `floors` | Number of floors |
| `waterfront`, `view`, `condition` | Property attributes |
| `yr_built`, `yr_renovated` | Year built / renovated |
| `city`, `country` | Location |

During preprocessing (`development.py`):
- `date` is split into `sale_day`, `sale_month`, `sale_year`
- `city` and `country` are label-encoded into numeric category codes

---

## 🚀 Step-by-Step Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify project files
Ensure these files are present in the root folder:
- `house_price_prediction.csv`
- `MLR.pkl` (pre-trained model — or retrain using the steps below)
- `templates/index.html`

---

## 🧠 Model Training

If you want to retrain the model from scratch instead of using the provided `MLR.pkl`:

```bash
python development.py
```

This script will:
1. Load `house price prediction.csv` (⚠️ update the filename inside `development.py` to match `house_price_prediction.csv` if needed)
2. Engineer date features (`sale_day`, `sale_month`, `sale_year`) and encode `city`/`country`
3. Split data into training (80%) and testing (20%) sets
4. Train a `LinearRegression` model
5. Print R² score and MSE for both train and test sets
6. Save the trained model as `MLR.pkl`

> **Important:** `app.py` expects `city` and `country` to be encoded using the fixed mappings defined in `CITY_MAPPING` and `COUNTRY_MAPPING`. If you retrain the model, confirm these mappings still match the encoding used during training.

---

## ▶️ Running the App Locally

### 1. Start the Flask server
```bash
python app.py
```

### 2. Open in browser
Navigate to:
```
http://127.0.0.1:5000/
```

### 3. Using gunicorn (production-style local run)
```bash
gunicorn app:app
```

### 4. Make a prediction
Fill in the form with property details (bedrooms, bathrooms, square footage, city, sale date, etc.) and submit to view the predicted price.

---

## ☁️ Deployment

This app is configured for deployment on platforms that support `Procfile`-based Python apps (e.g. **Heroku**, **Render**).

### General steps:
1. Push your code to a GitHub repository
2. Create a new web service/app on your hosting platform
3. Connect it to your GitHub repository
4. Set the build command to install `requirements.txt`
5. Ensure the start command matches the `procfile`:
   ```
   web: gunicorn app:app
   ```
6. Deploy and wait for the build to complete
7. Open the generated live URL to test the app

---

## 🌐 Live Demo

> 🔗 **Deployment Link:** _[Add your live deployment URL here once deployed]_


