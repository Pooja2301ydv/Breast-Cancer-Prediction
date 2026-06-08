"""
Evaluate the saved best model and produce plots/metrics under artifacts/.
Run after src/train.py.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_model, ensure_dir, save_json

RANDOM_STATE = 42

def main():
    ds = load_breast_cancer()
    X, y = ds.data, ds.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = load_model("models/best_model.joblib")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    ensure_dir("artifacts")
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).to_csv("artifacts/classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Malignant(0)", "Benign(1)"], rotation=45)
    plt.yticks(tick_marks, ["Malignant(0)", "Benign(1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("artifacts/roc_curve.png", dpi=160)
    plt.close()

    save_json({"roc_auc": float(roc_auc), "confusion_matrix": cm.tolist()}, "artifacts/eval_summary.json")
    print("Evaluation complete. See artifacts/.")

if __name__ == "__main__":
    main()
