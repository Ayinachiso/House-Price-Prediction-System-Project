from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Load the scaler (if used during training)
SCALER_PATH = "scaler.save"
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

@app.route("/")
def home():
    """Render the main page with the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive input from the form, preprocess, predict house price, and render result.
    """
    try:
        # Get form data
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        # Prepare input for model
        input_features = np.array([[area, bedrooms, bathrooms, stories, parking]])

        # Scale input if scaler exists
        if scaler:
            input_features = scaler.transform(input_features)

        # Predict price
        predicted_price = model.predict(input_features)[0][0]

        # Format result
        result = f"Estimated House Price: ${predicted_price:,.2f}"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

# Run the app
if __name__ == "__main__":
    # For local testing only; use gunicorn for deployment
    app.run(debug=True)