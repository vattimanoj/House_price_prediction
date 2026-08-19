import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, request, render_template

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
with open("MLR.pkl", "rb") as f:
    reg = pickle.load(f)

# ------------------------------------------------------------
# Feature order used while training the model
# ------------------------------------------------------------
FEATURE_ORDER = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "country",
    "sale_day",
    "sale_month",
    "sale_year",
]

# ------------------------------------------------------------
# City encoding
# IMPORTANT:
# These values MUST match your training data encoding.
# ------------------------------------------------------------
CITY_MAPPING = {
    "Algona": 0,
    "Auburn": 1,
    "Beaux Arts Village": 2,
    "Bellevue": 3,
    "Black Diamond": 4,
    "Bothell": 5,
    "Burien": 6,
    "Carnation": 7,
    "Clyde Hill": 8,
    "Covington": 9,
    "Des Moines": 10,
    "Duvall": 11,
    "Enumclaw": 12,
    "Fall City": 13,
    "Federal Way": 14,
    "Inglewood-Finn Hill": 15,
    "Issaquah": 16,
    "Kenmore": 17,
    "Kent": 18,
    "Kirkland": 19,
    "Lake Forest Park": 20,
    "Maple Valley": 21,
    "Medina": 22,
    "Mercer Island": 23,
    "Milton": 24,
    "Newcastle": 25,
    "Normandy Park": 26,
    "North Bend": 27,
    "Pacific": 28,
    "Preston": 29,
    "Ravensdale": 30,
    "Redmond": 31,
    "Renton": 32,
    "Sammamish": 33,
    "SeaTac": 34,
    "Seattle": 35,
    "Shoreline": 36,
    "Skykomish": 37,
    "Snoqualmie": 38,
    "Snoqualmie Pass": 39,
    "Tukwila": 40,
    "Vashon": 41,
    "Woodinville": 42,
    "Yarrow Point": 43,
}

# ------------------------------------------------------------
# Country encoding
# IMPORTANT:
# These values MUST match your training data encoding.
# ------------------------------------------------------------
COUNTRY_MAPPING = {
    "USA": 0,
}

# ------------------------------------------------------------
# Flask application
# ------------------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------------------
# Home page
# ------------------------------------------------------------
@app.route("/")
def main_page():
    return render_template("index.html")


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    try:
        form = request.form

        # ----------------------------------------------------
        # Convert date
        # ----------------------------------------------------
        sale_date = datetime.strptime(
            form["date"],
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # Get city name from HTML
        # ----------------------------------------------------
        city_name = form["city"]

        if city_name not in CITY_MAPPING:
            return render_template(
                "index.html",
                error=f"Invalid city: {city_name}"
            )

        city_code = CITY_MAPPING[city_name]

        # ----------------------------------------------------
        # Get country name from HTML
        # ----------------------------------------------------
        country_name = form["country"]

        if country_name not in COUNTRY_MAPPING:
            return render_template(
                "index.html",
                error=f"Invalid country: {country_name}"
            )

        country_code = COUNTRY_MAPPING[country_name]

        # ----------------------------------------------------
        # Create input row
        # ----------------------------------------------------
        row = {
            "bedrooms": float(form["bedrooms"]),
            "bathrooms": float(form["bathrooms"]),
            "sqft_living": float(form["sqft_living"]),
            "sqft_lot": float(form["sqft_lot"]),
            "floors": float(form["floors"]),
            "waterfront": int(form["waterfront"]),
            "view": int(form["view"]),
            "condition": int(form["condition"]),
            "sqft_above": float(form["sqft_above"]),
            "sqft_basement": float(form["sqft_basement"]),
            "yr_built": int(form["yr_built"]),
            "yr_renovated": int(form["yr_renovated"]),

            # Convert city name to encoded number
            "city": city_code,

            # Convert country name to encoded number
            "country": country_code,

            # Date features
            "sale_day": sale_date.day,
            "sale_month": sale_date.month,
            "sale_year": sale_date.year,
        }

        # ----------------------------------------------------
        # Create DataFrame in exact training order
        # ----------------------------------------------------
        features = pd.DataFrame(
            [row],
            columns=FEATURE_ORDER
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------
        prediction = reg.predict(features)[0]

        prediction_text = f"{prediction:,.2f}"

        # ----------------------------------------------------
        # Show result
        # ----------------------------------------------------
        return render_template(
            "index.html",
            prediction_text=prediction_text
        )

    except ValueError as e:
        return render_template(
            "index.html",
            error=f"Invalid input: {str(e)}"
        )

    except KeyError as e:
        return render_template(
            "index.html",
            error=f"Missing form field: {str(e)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


# ------------------------------------------------------------
# Run Flask application
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)