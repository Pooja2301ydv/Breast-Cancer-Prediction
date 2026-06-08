"""
Train two models (Logistic Regression & Random Forest) on Breast Cancer dataset,
select the best by ROC AUC, and save it to models/best_model.joblib.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
from pathlib import Path
from utils import save_model, save_json, ensure_dir

RANDOM_STATE = 42

def main():
    ds = load_breast_cancer()
    X, y = ds.data, ds.target
    feature_names = list(ds.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
    ])
    rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)

    models = {"logreg": logreg, "random_forest": rf}

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=-1)
        cv_results[name] = {"roc_auc_mean": float(np.mean(scores)), "roc_auc_std": float(np.std(scores))}

    test_scores = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        test_scores[name] = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "accuracy": float((y_pred == y_test).mean()),
            "f1": float(f1_score(y_test, y_pred))
        }
        fitted[name] = model

    best_name = max(test_scores, key=lambda n: test_scores[n]["roc_auc"])
    best_model = fitted[best_name]

    save_model(best_model, "models/best_model.joblib")

    # Save basic artifacts
    ensure_dir("artifacts")
    pd.DataFrame(X_train, columns=feature_names).describe().to_csv("artifacts/train_describe.csv", index=True)
    save_json({"cv_results": cv_results, "test_scores": test_scores, "best_model": best_name, "feature_names": feature_names}, "artifacts/metrics.json")

    print("Training complete. Best model:", best_name)

if __name__ == "__main__":
    main()
