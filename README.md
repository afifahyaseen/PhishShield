# **PhishShield: Machine Learning-Based Detection of Phishing Websites**

**Overview**

PhishShield is a command-line machine learning tool to detect phishing URLs.  
The repository contains code to train models, compare model performance, and run URL predictions. The primary production model used is Random Forest; other models (Logistic Regression, XGBoost) are included for comparison and evaluation.

**Installation**

1. Clone the repository
   
git clone https://github.com/<your-username>/PhishShield.git
cd PhishShield
2. Install base dependencies

pip install -r requirements.txt

**Note**: The scripts import xgboost and tabulate in some places:

-->If you plan to train or use the XGBoost model, install: pip install xgboost

-->If compare_models.py raises ModuleNotFoundError for tabulate, install: pip install tabulate

**Dataset format**

The dataset file dataset/phishshield_new_labeled_data.csv included in the repo has two columns:

url — string of the URL

label — integer label (0 = Legitimate, 1 = Phishing)

If you replace or extend this dataset, keep the same column names and format.

# 1. Train models from the dataset
(bash)
python scripts/train_model.py
Loads dataset/phishshield_new_labeled_data.csv, vectorizes URLs (TF-IDF), trains models (Random Forest, Logistic Regression, XGBoost), prints evaluation metrics, and saves model artifacts to models/ (e.g., random_forest_model.pkl, tfidf_vectorizer.pkl).

If you want only Random Forest for production, you can still run training and keep the Random Forest model file.
Model & Vectorizer files
tfidf_vectorizer.pkl and vectorizer.pkl are vectorizer artifacts used to transform raw URL text to numeric features. The prediction scripts load the appropriate vectorizer from models/ before transforming the input URL.

Core production model: Random Forest (random_forest_model.pkl).

Other models are present strictly for comparison/evaluation.
# 2. Compare model performance
(bash)
python scripts/compare_models.py
Loads saved models and performs evaluation on a test split. Prints accuracy, precision, recall, F1 (formatted with tabulate if available).

# 3. Run prediction with all models

(bash)
python scripts/predict_all.py

The script will prompt: Enter a URL to test against all models:

Output example format (one line per model):

Random Forest: Phishing 🚨
Logistic Regression: Legitimate ✅
XGBoost: Phishing 🚨
