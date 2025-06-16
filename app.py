from flask import Flask, request, jsonify, session, redirect, send_from_directory, render_template

from flask_cors import CORS
import pickle
import pandas as pd
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow cookies to be sent
app.secret_key = os.urandom(24)       # For session management

# In-memory user store for demo (can be replaced with a database)
users = {}

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the dataset and dynamically extract the symptom list
try:
    df = pd.read_csv("dataset.csv")
    symptom_list = [col.lower() for col in df.columns if col.lower() not in ['age', 'gender', 'disease', 'outcome variable']]
    logging.info("Dataset loaded successfully. Symptoms: %s", symptom_list)
except FileNotFoundError:
    symptom_list = []
    logging.error("Error: dataset.csv file not found.")

# Ensure the model is loaded, or terminate the app
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        logging.info("Label encoder loaded successfully.")

except FileNotFoundError as e:
    logging.critical(f"File not found: {e}")
    exit(1)
except Exception as e:
    logging.critical(f"Unexpected error during model loading: {e}")
    exit(1)

# ---- AUTH ROUTES ----
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if username in users:
        return jsonify({'error': 'Username already exists'}), 400
    users[username] = password
    session['user_id'] = username
    return jsonify({'message': 'Signup successful'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if users.get(username) == password:
        session['user_id'] = username
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/check_login', methods=['GET'])
def check_login():
    user_id = session.get('user_id')
    logging.info(f"Session check - Logged in user: {user_id}")
    return jsonify({'logged_in': bool(user_id)})

# ---- API ROUTES ----
@app.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": symptom_list})

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        logging.debug("Request Data: %s", request.json)

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        symptoms_data = data.get("symptoms")
        if not symptoms_data:
            return jsonify({"error": "'symptoms' is missing or empty"}), 400

        age = symptoms_data.get("age")
        gender = symptoms_data.get("gender")
        if age is None or gender is None:
            return jsonify({"error": "Both 'age' and 'gender' are required"}), 400

        for key in ['fever', 'cough', 'fatigue', 'difficulty breathing']:
           if key not in symptoms_data:
            return jsonify({"error": f"Missing symptom: {key}"}), 400
 

        # filtered_df = df[(df['Age'] <= int(age)) & (df['Gender'].str.lower() == str(gender).lower())]
        # if filtered_df.empty:
        #     return jsonify({"error": "No matching data found for the given age and gender."}), 404

        input_features = preprocess_input(symptoms_data)
        logging.debug(f"Input vector for prediction: {input_features}, Length: {len(input_features)}")

        probabilities = model.predict_proba([input_features])[0]
        disease_probabilities = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
        top_predictions = disease_probabilities[:3]

        predicted_diseases = [
            {"disease": label_encoder.inverse_transform([int(disease)])[0], "probability": f"{round(prob * 100, 2)}%"}
            for disease, prob in top_predictions
        ]

        if not predicted_diseases:
            return jsonify({"error": "No predictions could be made based on the input data."}), 404

        # ✅ Store input and predictions in session
        session['user_data'] = symptoms_data
        session['predicted_diseases'] = predicted_diseases

        return jsonify({"redirect": "https://diagnose-ai.onrender.com/prediction_result"})


    except KeyError as ke:
        logging.error(f"KeyError during prediction: {str(ke)}")
        return jsonify({"error": f"KeyError: {str(ke)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500



def preprocess_input(symptoms_data):
    try:
        # Convert input values to match model expectations
        fever = 1 if symptoms_data.get("fever", "").lower() == "yes" else 0
        cough = 1 if symptoms_data.get("cough", "").lower() == "yes" else 0
        fatigue = 1 if symptoms_data.get("fatigue", "").lower() == "yes" else 0
        breathing = 1 if symptoms_data.get("difficulty breathing", "").lower() == "yes" else 0

        age = int(symptoms_data.get("age", 0))
        gender = 1 if symptoms_data.get("gender", "").lower() == "male" else 0

        bp_map = {"low": 0, "normal": 1, "high": 2}
        chol_map = {"low": 0, "normal": 1, "high": 2}

        bp = bp_map.get(symptoms_data.get("blood pressure", "").lower(), 1)
        chol = chol_map.get(symptoms_data.get("cholesterol level", "").lower(), 1)

        input_vector = [fever, cough, fatigue, breathing, age, gender, bp, chol]

        if len(input_vector) != 8:
            raise ValueError("Invalid input length")

        return input_vector

    except Exception as e:
        raise ValueError(f"Error processing input: {e}")

# ---- FRONTEND ROUTES ----

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

@app.route('/frontpage')
def serve_frontpage():
    return send_from_directory('frontend', 'index.html')

@app.route('/home')
def serve_home():
    return send_from_directory('frontend', 'Home.html')

@app.route('/login')
def serve_login():
    return send_from_directory('frontend', 'login.html')

@app.route('/signup')
def serve_signup():
    return send_from_directory('frontend', 'signup.html')

@app.route('/aboutus')
def serve_aboutus():
    return send_from_directory('frontend', 'aboutus.html')

@app.route('/confirmation')
def serve_confirmation():
    return send_from_directory('frontend', 'confirmation.html')

@app.route('/contactus')
def serve_contactus():
    return send_from_directory('frontend', 'contactus.html')

@app.route('/learnmore')
def serve_learnmore():
    return send_from_directory('frontend', 'learnmore.html')

@app.route('/prediction')
def serve_prediction():
    return send_from_directory('frontend', 'prediction.html')

@app.route('/test')
def serve_test():
    return send_from_directory('frontend', 'test.html')

# This one uses Jinja template rendering (keep this with `render_template`)
@app.route('/prediction_result')
def prediction_result():
    predictions = session.get('predicted_diseases', [])
    return render_template('prediction_result.html', predictions=predictions)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

