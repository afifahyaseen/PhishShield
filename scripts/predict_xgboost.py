import os
import joblib

test_url = input("Enter a URL to test using XGBoost: ")

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

url_vec = vectorizer.transform([test_url])
prediction = model.predict(url_vec)[0]

label = "Legitimate ✅" if prediction == 0 else "Phishing 🚨"
print(f"XGBoost Prediction: {label}")
