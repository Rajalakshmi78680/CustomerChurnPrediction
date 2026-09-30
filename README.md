# CustomerChurnPrediction
Customer Churn Prediction
# AI-Powered Telecom Customer Churn Prediction & Retention Analytics
*Machine Learning + Explainable AI + Business Intelligence*

## Verification status (read first)
| Part | Status |
|---|---|
| Data validation, cleaning, feature engineering, preprocessing | Built and run on the real 7,043-row IBM Telco workbook |
| Training, CV, tuning, threshold, calibration, fairness, saved models | Built and run; results in `outputs/` |
| Prediction, validation, recommendations, permutation importance | Built and run on example customers |
| Tests (`tests/`) | 15 tests pass using a stand-in runner. Real `pytest` was not installable (no network) |
| Streamlit app (`app/app.py`) | Written; syntax-checked only. **Never launched** (streamlit/plotly not installed) |
| SHAP | Code path written, **not executed**. A labelled fallback was run instead |
| XGBoost, imblearn (SMOTE) | Not available in the build environment, so not used |

## Problem
Telecom operators lose revenue when customers leave. Predict who is likely to churn so retention offers can be targeted.

## Dataset
IBM Telco Customer Churn, 7,043 customers, 26.5% churn. The provided `Telco_customer_churn.xlsx` has renamed headers (handled in `data_loader.RENAME`) and
extra columns. **Excluded as leakage:** `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`. Geography columns and `customerID` are not features.
Blank `TotalCharges` (11 rows) all have tenure 0 and are set to 0.

## Pipeline
Validate -> clean -> engineer features -> stratified 80/20 split (seed 42) -> ColumnTransformer (median+scale / mode+one-hot, fitted on train only)
-> RandomizedSearchCV (12 iterations, 5-fold stratified, scoring PR-AUC) for 5 models -> select by mean CV PR-AUC (training data only)
-> threshold chosen on out-of-fold training predictions (max F1 over 0.20-0.60) -> sigmoid calibration kept only if test Brier improves -> save.

**Class imbalance:** handled with `class_weight="balanced"` as a tuned option (SMOTE lives in imbalanced-learn, unavailable here). Test-set numbers are never used for selection.

**Engineered features:** TenureGroup, AverageMonthlySpend (falls back to MonthlyCharges at tenure 0), ServiceCount, HasSecurityService, HasStreamingService, IsLongTermCustomer, IsHighValueCustomer (fixed $80 cutoff).

## Run
```bash
pip install -r requirements.txt
python src/train.py          # trains, writes models/ and outputs/
streamlit run app/app.py
python -m pytest tests
```

## Explainability
`src/explainability.py` uses SHAP if installed. Otherwise: permutation importance (global) and one-feature-at-a-time baseline substitution (single customer).
Both describe model behaviour, not causes. Baseline substitution ignores feature interactions, so it is only an approximation.

## Limitations
No temporal split (the data has no dates); one hold-out set of 1,409 customers; dashboard probabilities for all customers include training rows (optimistic);
monitoring uses the reference data only; business simulator is assumption-driven; fairness tables are descriptive, not a fairness verdict.
Senior citizens show a higher false-positive rate than non-seniors at the chosen threshold.

## Screenshots
_(placeholders: add after running the app)_

## Team contributions
_(fill in)_
