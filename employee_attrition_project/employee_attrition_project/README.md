# Employee Attrition Prediction and Workforce Retention Analysis using Machine Learning

 AI/ML-Based Real-World Application

**Group members:** _(TEITA30  Sanket Kandalkar
TEITA70    Avdhoot dhamdhere 
TEITA13   Keshav Chaudhari
TEITA07     Vijay Bhosale)_

## Problem Statement
Replacing an employee who leaves can cost several months of their salary, yet HR teams usually learn about resignations only after they happen. This project builds a machine learning system that **predicts which employees are at risk of leaving, explains why, groups employees into risk tiers and segments, and estimates the value of acting on those predictions**, so retention effort goes to the right people.

## Dataset
**IBM HR Analytics Employee Attrition & Performance** (Kaggle): 1,470 employees, 35 columns, target `Attrition` (Yes/No).
Download: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Place the file at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.
> If the file is missing, the notebook generates a *simulated* dataset with the same columns so that the code still runs. Always use the real CSV for final results and say so in the report.

## What the Project Does
1. **EDA and retention analysis**: attrition by overtime, role, department, travel, tenure, pay and promotion gap
2. **Feature engineering**: satisfaction index, income per experience year, tenure ratio, job-hopping indicator, promotion stagnation
3. **Feature selection**: mutual-information ranking and `SelectKBest` tuned by cross-validation
4. **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, with class weights for imbalance
5. **Honest model selection**: tuned with 5-fold CV, chosen by cross-validated PR-AUC, test set used only once at the end
6. **Probability calibration** so that risk scores behave like real probabilities
7. **Business-driven threshold**: chosen from replacement cost, retention-action cost and success rate (editable assumptions) rather than the default 0.5
8. **Explainability**: permutation importance and logistic-regression odds ratios; top risk drivers per employee
9. **Risk tiers** (Low / Medium / High) for every employee and a downloadable high-risk watch-list
10. **Employee segmentation** with K-Means
11. **Fairness audit** by gender, marital status and age group (sensitive attributes are *not* used as model inputs)
12. **Business impact** comparison: do nothing vs talk to everyone vs use the model
13. **Retention recommendations** linked to the main drivers, and a `predict_employee()` function

## Repository Structure
```
.
├── employee_attrition_retention.ipynb   # full project notebook
app.py
├── requirements.txt
├── README.md
├── data/                                 # put the Kaggle CSV here
├── images/                               # plots saved when the notebook runs
├── models/                               # trained model saved when the notebook runs
└── reports/   
app.py                           # high_risk_employees.csv created when the notebook runs
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook employee_attrition_retention.ipynb
```
Run all cells from top to bottom. The notebook prints which data source it used at the start.

## Results
> **Important:** the numbers below come from a test run on the notebook's **simulated** sample data (the real Kaggle CSV was not available when this was built). They only show what the output looks like. **Download the real CSV, put it in `data/`, re-run the notebook, and replace these numbers and the images in `images/` with your own before submitting.**

Sample run (simulated data, 1,470 employees, about 16% leave):

| Item | Sample-run value |
|---|---|
| Final model | Logistic Regression (tuned, calibrated) |
| Test ROC-AUC / PR-AUC | 0.83 / 0.48 (a random model scores 0.16 PR-AUC) |
| Business threshold | 0.30 (theoretical break-even is about 0.33) |
| Recall / precision at threshold | 53% / 50% |
| Actual attrition: Low / Medium / High tier | 5% / 24% / 47% |

The risk tiers separate employees well: actual attrition climbs steeply from the Low to the High tier. The notebook also prints a business-impact table (do nothing vs talk to everyone vs use the model) and a fairness audit.

## Limitations and Ethics
- The IBM dataset is a public fictional benchmark; real company data will behave differently.
- The model shows association, not causation. Pilot any retention action and measure its effect.
- Business cost assumptions are illustrative and must be replaced with real figures.
- Predictions should support supportive HR conversations only, never punishment, hiring or firing decisions.

## Future Work
Survival analysis (time to exit), SHAP explanations, a Streamlit dashboard for HR managers, drift monitoring and periodic retraining.
