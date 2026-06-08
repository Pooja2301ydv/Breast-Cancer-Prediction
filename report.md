# Industrial Training Report — Machine Learning with Data Science
**Title:** Breast Cancer Diagnosis Prediction using ML  
**Student:** _Your Name_ · **Department:** _Your Department_ · **Year:** 4th Year  
**Institute:** _Your College_ · **Guide/HOD:** _HOD Name_  
**Duration:** 6–8 weeks (example) · **Tools:** Python, scikit-learn, Streamlit, VS Code

---

## Abstract
This project builds an end-to-end ML pipeline to predict whether a breast tumor is malignant (0) or benign (1). Two algorithms—Logistic Regression and Random Forest—are trained and compared. The best model is selected by ROC AUC and deployed via a Streamlit app for interactive predictions.

## Introduction
- Problem context and importance of early detection  
- Project objective: demonstrate a full workflow from data to deployment

## Dataset
- Source: `sklearn.datasets.load_breast_cancer` (569 samples, 30 features)  
- Target: 0 = Malignant, 1 = Benign  
- Split: 80% train / 20% test (stratified, random_state=42)

## Methodology
1. Preprocessing with `StandardScaler` for LR via `Pipeline`
2. Models: Logistic Regression & Random Forest (baseline settings)
3. Model selection: 3-fold cross-validation by ROC AUC
4. Evaluation: AUC, Accuracy, F1, Confusion Matrix, ROC curve
5. Deployment: `streamlit` app (`app.py`) that loads `models/best_model.joblib`

## Experiments & Results (Fill after running code)
- From `artifacts/metrics.json`, copy:
  - Cross-Validation ROC AUC (mean ± std) for both models
  - Test AUC, Accuracy, F1 for both models
  - Best model name
- Include screenshots of:
  - `artifacts/roc_curve.png`
  - `artifacts/confusion_matrix.png`
  - Streamlit app prediction

## Discussion
- Interpretation of metrics and errors  
- Trade-offs between LR and RF  
- Suggested improvements: hyperparameter tuning, feature importance, calibration, explainability (e.g., SHAP)

## Conclusion
Summary of achievements, limitations, and future scope.

## How to Run (for HOD)
1. Create a virtual environment and install requirements
2. `python src/train.py`
3. `python src/evaluate.py`
4. `streamlit run app.py`

## References
- Scikit-learn documentation & dataset description
