import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "phishshield_new_labeled_data.csv")
df = pd.read_csv(DATASET_PATH)

X = df["url"]
y = df["label"]

# Vectorize URLs
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Save vectorizer
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Define models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\n=== Training {name.upper()} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(model, os.path.join(models_dir, f"{name}_model.pkl"))

print("\n✅ All models trained and saved.")
