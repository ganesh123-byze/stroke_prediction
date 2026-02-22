import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/stroke_data.csv"

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop("id", axis=1)

X = df.drop("stroke", axis=1)

# Take small sample
X_sample = X.sample(200, random_state=42)

# Separate pipeline steps
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Transform data
X_processed = preprocessor.transform(X_sample)

print("Building SHAP explainer...")

# If tree-based model
if classifier.__class__.__name__ in ["RandomForestClassifier", "GradientBoostingClassifier"]:
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)

    shap.summary_plot(shap_values[1], X_processed, show=False)

# If logistic regression
elif classifier.__class__.__name__ == "LogisticRegression":
    explainer = shap.LinearExplainer(classifier, X_processed)
    shap_values = explainer.shap_values(X_processed)

    shap.summary_plot(shap_values, X_processed, show=False)

else:
    raise ValueError("Model type not supported for SHAP explanation.")

plt.tight_layout()
plt.savefig("models/shap_summary.png")

print("SHAP summary plot saved successfully.")