# Titanic Analytics EDA

This notebook script performs Part A profiling, cleaning, and data story analysis for the Titanic dataset.

## What is covered

- Loaded the dataset once with `sns.load_dataset('titanic')`.
- Saved the loaded raw dataset to `analytics/titanic.csv` for offline grading.
- Calculated missing-value percentages for every column with missing data.
- Applied threshold-based missing-value handling:
  - <5% missing → drop rows
  - 5%–30% missing → impute
  - >30% missing → drop or encode "Missing" with justification
- Created univariate plots and outlier counts for `age` and `fare`.
- Computed survival rates by `sex`, `pclass`, and `sex`+`pclass` using boolean masking.
- Built a 6×6 correlation matrix restricted to `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`.
- Generated a multivariate data story with 4 charts.
- Standardized `age` and `fare` as an exploratory sanity check.

## Missing-Value Decisions

- `deck` has 77.22% missing values, so it is dropped because it cannot be reliably imputed and it is not essential to the core profiling analysis.
- `age` has 19.87% missing values, so it is imputed with the median because numeric imputation is reasonable at this rate.
- `embarked` has 0.22% missing values, so the missing rows are dropped because the rate is low and row removal retains data quality.

## Top Correlation Findings

- The strongest correlation is between `fare` and `pclass`.
- The second strongest correlation is between `parch` and `sibsp`.

## Multivariate Story Charts

1. Survival by Sex: females were much more likely to survive than males. This plot highlights the strong gender gap in survival outcomes, with female survival rates far exceeding male rates. It supports the narrative that sex was a primary factor in who lived.

2. Survival by Pclass: first-class passengers had the highest survival rate, while third-class had the lowest. The chart shows that economic status and ticket class were closely tied to survival, suggesting that better accommodation and access to lifeboats mattered. It reinforces the idea that class privilege influenced outcomes.

3. Age vs Survival boxplot: survivors tend to be younger, showing age-related survival differences. The boxplot makes it clear that non-survivors have a higher median age and more older passengers in the upper tail. This supports the data story that younger passengers were more likely to be prioritized or able to survive.

4. Fare vs Age scatterplot colored by survival: higher fare and younger age are both associated with greater survival probability. The scatterplot reveals that people who paid more and were younger are more often in the surviving group, while lower fare third-class passengers cluster among non-survivors. This chart ties together economic and demographic factors in the survival story.

## Standardization Check

- `age` and `fare` are transformed using z-scores.
- The script prints pre- and post-standardization means and standard deviations to confirm the transformed columns have mean ≈ 0 and std ≈ 1.

## Part B — Modeling

- `modeling.py` loads the committed `analytics/titanic.csv` and continues from the same cleaned dataset.
- It performs a stratified train/test split on `survived`, preserving class balance in both splits.
- It preprocesses training data only with a `ColumnTransformer` for numeric imputation/scaling and categorical encoding.
- It trains and compares Logistic Regression, Decision Tree, and Random Forest classifiers.
- It evaluates classifiers using confusion matrix, accuracy, precision, recall, F1, and ROC AUC.
- It compares imbalance handling strategies: baseline, `class_weight='balanced'`, and SMOTE oversampling on training data.
- It tunes Random Forest hyperparameters with `GridSearchCV` and reports the best parameters plus OOB score.
- It trains a linear regression model for predicting `fare` and reports MAE, RMSE, R², and Adjusted R².
- It saves the complete end-to-end pipeline to `analytics/best_pipeline.joblib` and reloads it to confirm predictions on raw input.

## Artifacts

- `analytics/titanic.csv`: committed offline fallback produced by `eda.py`
- `analytics/best_pipeline.joblib`: saved end-to-end sklearn pipeline including preprocessing and final estimator
- `analytics/plots/roc_curve_comparison.png`, `analytics/plots/decision_tree.png`, `analytics/plots/regression_residual_plot.png`: saved model and regression plots
- `analytics/eda.py`: Part A profiling, cleaning, and data story pipeline
- `analytics/modeling.py`: Part B modeling pipeline that loads `analytics/titanic.csv` and does not call `sns.load_dataset('titanic')`

## Key Results

- Tuned Random Forest is recommended for deployment with test accuracy ≈ 0.816, AUC ≈ 0.843, precision ≈ 0.875, recall ≈ 0.609, and F1 ≈ 0.718.
- SMOTE oversampling achieved the best F1 score among imbalance strategies while maintaining a balanced precision/recall tradeoff.
- Regression prediction of `fare` achieved MAE ≈ 20.81, RMSE ≈ 30.47, R² ≈ 0.400, and Adjusted R² ≈ 0.368.
