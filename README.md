# Machine Learning Industrial Training Project (VS Code Ready)

End-to-end ML project that trains a classifier on the **Breast Cancer Wisconsin** dataset (bundled with scikit-learn), evaluates it, and serves predictions via a **Streamlit** app.

## Setup & Run (Windows/Mac/Linux)

```bash
# 1) Open this folder in VS Code

# 2) Create & activate virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Train (saves best model to models/best_model.joblib)
python src/train.py

# 5) Evaluate (saves plots & metrics to artifacts/)
python src/evaluate.py

# 6) Run demo app for HOD presentation
streamlit run app.py
```

## What to Show in Front of HOD
1. Open the repo in VS Code; show code structure and comments.
2. Run `python src/train.py` and point to the saved model and `artifacts/metrics.json`.
3. Run `python src/evaluate.py` and open `artifacts/roc_curve.png` and `artifacts/confusion_matrix.png`.
4. Launch `streamlit run app.py` and do:
   - Manual single prediction (enter numbers)
   - CSV batch prediction (show columns printed by the app and upload a sample file if you create one)

## Notes
- Dataset is offline (from scikit-learn), so it works without internet.
- Reproducible with `random_state=42` and a fixed train/test split.
