import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "phishshield_new_labeled_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load dataset
df = pd.read_csv(DATASET_PATH)
X = df["url"]
y = df["label"]

# Load vectorizer and transform data
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
X_vec = vectorizer.transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model files
model_files = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Evaluate models
results = []

for name, filename in model_files.items():
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        continue

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

# Print table
headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
print("\n📊 Model Comparison:\n")
print(tabulate(results, headers=headers, floatfmt=".4f"))

