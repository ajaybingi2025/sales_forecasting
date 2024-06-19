from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('home.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from the request
            date_to_predict = request.form['Date']
            product_id = int(request.form['product ID'])
            n_samples_per_day = int(request.form['n_samples_per_day'])

            print(f"Input data: Date={date_to_predict}, Product ID={product_id}, Samples per Day={n_samples_per_day}")

            # Convert date to numerical format
            date_numeric = datetime.strptime(date_to_predict, "%Y-%m-%d").timestamp()

            # Assuming product_type is categorical, you may need to encode it
            samples = 1782

            # Prepare input data
            X = np.array([date_numeric, product_id, n_samples_per_day])
            X = np.append(X, [samples])
            X_reshaped = X.reshape(1, 1, len(X))
            X_padded = np.pad(X_reshaped, ((0, 0), (0, 3), (0, 1782 - len(X))), 'constant', constant_values=0)
            reshaped_data = X_padded.tolist()

            print(f"Processed input data: {reshaped_data}")

            # Make prediction
            prediction = model.predict(np.array(reshaped_data))

            print(f"Raw prediction: {prediction}")

            # Extract the average value from the first dimension
            prediction_array = np.mean(prediction[0, 0, :]) * 1000

            print(f"Final prediction: {prediction_array}")

            # Redirect to 'pass.html' with the prediction value
            return render_template('pass.html', prediction=prediction_array)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"Prediction error: {error_message}")

            # Return error message if prediction fails
            return render_template('pass.html', prediction=None, error=error_message)

if __name__ == 'main':
    app.run(debug=True, use_reloader=False)