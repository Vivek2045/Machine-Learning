import os
import pandas as pd
import joblib
from flask import Flask, render_template, request

# Specify the path to the templates folder explicitly
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))


# Define the model path (adjust if necessary)
model_path = 'model/house_price_model.pkl'  # Update the path if your model is in a subfolder

# Check if the model file exists and load it
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
else:
    model = None
    print(f"Model file '{model_path}' not found!")

# Load the training dataset to get the feature columns used in training
data = pd.read_csv('house_dataset.csv')
feature_columns = data.drop('SalePrice', axis=1).columns

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Retrieve form data
            overall_qual = float(request.form["OverallQual"])
            gr_liv_area = float(request.form["GrLivArea"])
            year_built = float(request.form["YearBuilt"])
            total_bsmt_sf = float(request.form["TotalBsmtSF"])
            first_flr_sf = float(request.form.get("1stFlrSF", 0))  # Default to 0 if not provided
            second_flr_sf = float(request.form.get("2ndFlrSF", 0))  # Default to 0 if not provided
            bedroom_abv_gr = float(request.form.get("BedroomAbvGr", 0))  # Default to 0 if not provided

            # Check if the model is loaded
            if model is None:
                prediction = "Error: Model not loaded correctly!"
            else:
                # Prepare features for prediction (ensure all columns are included)
                features = pd.DataFrame([{
                    'OverallQual': overall_qual,
                    'GrLivArea': gr_liv_area,
                    'YearBuilt': year_built,
                    'TotalBsmtSF': total_bsmt_sf,
                    '1stFlrSF': first_flr_sf,
                    '2ndFlrSF': second_flr_sf,
                    'BedroomAbvGr': bedroom_abv_gr,
                    # Add more features as required
                }])

                # Add missing columns with default value of 0
                for col in feature_columns:
                    if col not in features.columns:
                        features[col] = 0

                # Reorder the columns to match the model's expected order
                features = features[feature_columns]

                # Make prediction
                prediction_value = model.predict(features)[0]

                # Format the prediction to a readable output
                prediction = f"Predicted House Price: ${prediction_value:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
