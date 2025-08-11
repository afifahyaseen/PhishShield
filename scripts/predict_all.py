import os
import joblib

# Ask for test URL
test_url = input("Enter a URL to test against all models: ")

# Set model + vectorizer paths
base_dir = os.path.dirname(__file__)
model_dir = os.path.join(base_dir, "..", "models")

model_files = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

url_vec = vectorizer.transform([test_url])

print("\n--- Model Predictions ---")
for model_name, filename in model_files.items():
    model_path = os.path.join(model_dir, filename)
    model = joblib.load(model_path)

    prediction = model.predict(url_vec)[0]
    label = "Legitimate ✅" if prediction == 0 else "Phishing 🚨"

    print(f"{model_name}: {label}")
