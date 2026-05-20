# Hospital Readmission Prediction - Machine Learning

## Overview

After building a REST API and React dashboard to explore historical hospital readmission data, I wanted to take it a step further and extend it from "here's what happened" to "here's what's likely to happen". This machine learning extension predicts the likelihood of patient readmission using ten years of diabetic patient data across 130 US hospitals (1999-2008), given a patient's clinical profile. The model can predict readmission probability for an individual patient given their clinical features.

## Related Projects

- **REST API**: [github.com/michellely825/Hospital-Readmissions-Api](https://github.com/michellely825/Hospital-Readmissions-Api)
- **React Dashboard**: [github.com/michellely825/Hospital-Readmission-Dashboard](https://github.com/michellely825/Hospital-Readmission-Dashboard)

## Tech-Stack

- **Language**: Python
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn
- **Environment**: Jupyter Notebook

## Dataset

- **Source**: [Kaggle - Hospital Readmissions](https://www.kaggle.com/datasets/dubradave/hospital-readmissions/data)
- **Records**: 25,000 patient encounters
- **Time Period**: 1999-2008
- **Features**: age, time in hospital, diagnoses, medications, lab procedures, readmission status and more

## Project Workflow

1. **Exploratory Data Analysis** - investigated distributions, readmission rates, and relationships between features
2. **Preprocessing** - encoded categorical and binary variables, prepared data for modeling
3. **Train/Test Split** - 80/20 stratified split to preserve the 53/47 readmission balance
4. **Modeling** - trained and compared logistic regression and random forest models
5. **Evaluation** — assessed performance using accuracy, classification report, ROC-AUC, and confusion matrix
6. **Single Patient Prediction** - model outputs readmission probability for an individual patient

## Results

| Model                   | ROC-AUC    |
| ----------------------- | ---------- |
| Logistic Regression     | 0.6459     |
| Random Forest (default) | 0.6357     |
| Random Forest (tuned)   | **0.6600** |

The tuned random forest was the best performing model. The default random forest showed clear overfitting with 100% training accuracy but only 60.64% test accuracy but tuning by limiting tree depth and minimum samples per split resolved this overfitting issue.

## Limitations & Next Steps

- ROC-AUC of 0.66 indicates moderate predictive ability so there is room for improvement
- The model misses a significant portion of at-risk patients (low recall for readmitted class)
- Future improvements could include more feature engineering, XGBoost, or SHAP explainability to interpret predictions for clinicians
