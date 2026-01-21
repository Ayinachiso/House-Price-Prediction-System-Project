
# app.py
# Flask web app for House Price Prediction
# Uses scikit-learn model and matches the six features in the form

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (scikit-learn pipeline)
MODEL_PATH = "model/house_price_model.pkl"
model = joblib.load(MODEL_PATH)


# Home route: render the main page with the input form
@app.route("/")
def home():
    return render_template("index.html")


# Predict route: receive input, preprocess, predict, and render result

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data for the six model features
        overallqual = int(request.form["OverallQual"])
        grlivarea = float(request.form["GrLivArea"])
        totalbsmtsf = float(request.form["TotalBsmtSF"])
        garagecars = int(request.form["GarageCars"])
        bedroomabvgr = int(request.form["BedroomAbvGr"])
        neighborhood = request.form["Neighborhood"].strip()


        # Prepare input for model as a DataFrame with correct columns
        input_features = pd.DataFrame([{
            'OverallQual': overallqual,
            'GrLivArea': grlivarea,
            'TotalBsmtSF': totalbsmtsf,
            'GarageCars': garagecars,
            'BedroomAbvGr': bedroomabvgr,
            'Neighborhood': neighborhood
        }])

        # Predict price using the loaded pipeline
        predicted_price = model.predict(input_features)[0]

        # Format result for display (rounded, comma separated)
        result = f"{predicted_price:,.0f}"
        # If AJAX/JS request, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'result': result})
    except Exception as e:
        result = f"Error: {str(e)}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'result': result})

    # Fallback: render template for normal POST
    return render_template("index.html", result=result)


# Run the app (for local testing only; use gunicorn or similar for deployment)
if __name__ == "__main__":
    app.run(debug=True)