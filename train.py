import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    f1_score,
    precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
# ==============================
DATA_PATH = "data/stroke_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================
# LOAD DATA
# ==============================
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

# ==============================
# PREPARE DATA
# ==============================
def prepare_data(df: pd.DataFrame):

    if "stroke" not in df.columns:
        raise ValueError("Target column 'stroke' not found.")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

# ==============================
# BUILD PREPROCESSOR
# ==============================
def build_preprocessor(X: pd.DataFrame):

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

# ==============================
# CROSS VALIDATION
# ==============================
def cross_validate_model(pipeline, X_train, y_train, model_name):

    logging.info(f"\nCross-validating {model_name}...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    roc_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=skf, scoring="roc_auc"
    )

    recall_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=skf, scoring="recall"
    )

    logging.info(f"Mean ROC-AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
    logging.info(f"Mean Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")

# ==============================
# TEST EVALUATION
# ==============================
def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logging.info(f"\n===== {model_name} Test Evaluation =====")
    logging.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"Test ROC-AUC: {roc_auc:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1: {f1:.4f}")

    return roc_auc

# ==============================
# THRESHOLD TUNING
# ==============================
def tune_threshold(model, X_test, y_test):

    logging.info("\nThreshold Tuning (Medical Optimization)")
    logging.info("-" * 50)

    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.1)

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info(f"Threshold: {threshold:.1f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info("-" * 30)

# ==============================
# MAIN FUNCTION
# ==============================
def main():

    try:
        logging.info("Starting training pipeline...")

        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = prepare_data(df)
        preprocessor = build_preprocessor(X_train)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE)
        }

        best_score = 0
        best_model = None
        best_model_name = ""

        for name, model in models.items():

            logging.info(f"\nTraining {name}...")

            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", model)
            ])

            # Cross-validation
            cross_validate_model(pipeline, X_train, y_train, name)

            # Fit full training data
            pipeline.fit(X_train, y_train)

            # Evaluate on test set
            score = evaluate_model(pipeline, X_test, y_test, name)

            # Threshold tuning
            tune_threshold(pipeline, X_test, y_test)

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_model_name = name

        logging.info(f"\nBest Model: {best_model_name}")
        logging.info(f"Best Test ROC-AUC: {best_score:.4f}")

        joblib.dump(best_model, MODEL_PATH)
        logging.info("Best model saved successfully.")

        logging.info("Training completed successfully.")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()