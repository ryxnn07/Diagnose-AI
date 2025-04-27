import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def preprocess_data(df):
    """Preprocess the dataset to handle categorical variables and missing values."""
    try:
        # Convert categorical columns to numeric (if applicable)
        if 'Fever' in df.columns:
            df['Fever'] = df['Fever'].map({'Yes': 1, 'No': 0})
        if 'Cough' in df.columns:
            df['Cough'] = df['Cough'].map({'Yes': 1, 'No': 0})
        if 'Fatigue' in df.columns:
            df['Fatigue'] = df['Fatigue'].map({'Yes': 1, 'No': 0})
        if 'Difficulty Breathing' in df.columns:
            df['Difficulty Breathing'] = df['Difficulty Breathing'].map({'Yes': 1, 'No': 0})
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        if 'Blood Pressure' in df.columns:
            df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
        if 'Cholesterol Level' in df.columns:
            df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})

        # Drop rows with missing values
        df = df.dropna()

        return df
    except KeyError as e:
        logging.error(f"Missing column during preprocessing: {e}")
        raise

# ---- Step 1: Load the Dataset ----
try:
    df = pd.read_csv("dataset.csv")
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Error: dataset.csv file not found.")
    exit()

# ---- Step 2: Preprocess the Data ----
try:
    df = preprocess_data(df)

    # ✅ Step 2.1: Encode the target column
    label_encoder = LabelEncoder()
    df["Outcome Variable"] = label_encoder.fit_transform(df["Disease"])

    logging.info("Data preprocessing and encoding completed successfully.")
except Exception as e:
    logging.error(f"Error during preprocessing: {e}")
    exit()

# ---- Step 3: Split Features and Target ----
try:
    X = df.drop(["Outcome Variable", "Disease"], axis=1)
    y = df["Outcome Variable"]

    print("Feature names:", X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")
except KeyError as e:
    logging.error(f"Missing target column 'Outcome Variable': {e}")
    exit()

# ---- Step 4: Train the Model ----
try:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed successfully.")
except Exception as e:
    logging.error(f"Error during model training: {e}")
    exit()

# ---- Step 5: Evaluate the Model ----
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy on Test Set: {accuracy:.2f}")
except Exception as e:
    logging.error(f"Error during model evaluation: {e}")
    exit()

# ---- Step 6: Save the Model and Encoder ----
try:
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
        logging.info("Model saved as model.pkl.")

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
        logging.info("Label encoder saved as label_encoder.pkl.")
except Exception as e:
    logging.error(f"Error saving the model or encoder: {e}")
