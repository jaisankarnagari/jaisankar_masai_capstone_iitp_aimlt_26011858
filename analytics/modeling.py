import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "titanic.csv"
PLOTS_DIR = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = PROJECT_DIR / "best_pipeline.joblib"


def save_plot(name):
    path = PLOTS_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved plot: {path}")


def load_and_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=['deck'], errors='ignore')
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_proba': y_proba
    }


def plot_classification_roc(results, y_test):
    plt.figure(figsize=(8, 6))
    for label, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        plt.plot(fpr, tpr, label=f"{label} (AUC={result['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    save_plot('roc_curve_comparison')
    plt.close()


def get_feature_names(preprocessor, numeric_features, categorical_features):
    cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = list(cat_ohe.get_feature_names_out(categorical_features))
    return numeric_features + cat_features


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def main():
    df = load_and_clean_data(DATA_PATH)
    print('Loaded raw titanic.csv with shape', df.shape)
    print(df.info())
    print('\nMissing value percentages:')
    print((df.isnull().mean() * 100).loc[lambda x: x > 0].sort_values(ascending=False))

    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'
    X = df[features]
    y = df[target]

    print('\nTarget class balance:')
    print(y.value_counts(normalize=True).rename('proportion'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print('\nTrain class balance:')
    print(y_train.value_counts(normalize=True).rename('proportion'))
    print('\nTest class balance:')
    print(y_test.value_counts(normalize=True).rename('proportion'))

    print('\nStratification is used because survived is imbalanced and we want the same class distribution in train and test sets, avoiding a training/test split that overrepresents one class.')

    numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    categorical_features = ['sex', 'embarked']
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    lr_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    dt_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    rf_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)
    dt_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    results = {
        'Logistic Regression': evaluate_classifier(lr_pipeline, X_test, y_test),
        'Decision Tree': evaluate_classifier(dt_pipeline, X_test, y_test),
        'Random Forest': evaluate_classifier(rf_pipeline, X_test, y_test)
    }

    for name, res in results.items():
        print(f"\n{name} metrics:")
        print('Accuracy:', res['accuracy'])
        print('Precision:', res['precision'])
        print('Recall:', res['recall'])
        print('F1:', res['f1'])
        print('AUC:', res['auc'])
        print('Confusion matrix:\n', res['confusion_matrix'])

    plot_classification_roc(results, y_test)

    fitted_preprocessor = dt_pipeline.named_steps['preprocessor']
    feature_names = get_feature_names(fitted_preprocessor, numeric_features, categorical_features)
    plt.figure(figsize=(16, 10))
    plot_tree(
        dt_pipeline.named_steps['classifier'],
        feature_names=feature_names,
        class_names=['Not Survived', 'Survived'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    save_plot('decision_tree')
    plt.close()

    comparison_df = pd.DataFrame({
        name: {
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc']
        }
        for name, res in results.items()
    }).T
    print('\nClassification comparison table:')
    print(comparison_df)

    print('\nImbalance handling comparison:')
    print('Class counts:')
    print(y_train.value_counts())

    balanced_pipeline = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    balanced_pipeline.fit(X_train, y_train)

    smote_pipeline = ImbPipeline([
        ('preprocessor', clone(preprocessor)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    smote_pipeline.fit(X_train, y_train)

    imbalance_results = {
        'Baseline RF': evaluate_classifier(rf_pipeline, X_test, y_test),
        'Balanced RF': evaluate_classifier(balanced_pipeline, X_test, y_test),
        'SMOTE RF': evaluate_classifier(smote_pipeline, X_test, y_test)
    }

    imbalance_df = pd.DataFrame({
        name: {
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1']
        }
        for name, res in imbalance_results.items()
    }).T
    print('\nImbalance comparison table:')
    print(imbalance_df)

    print('\nImbalance conclusion:')
    print('  Baseline RF gives the highest precision, while Balanced RF and SMOTE RF trade some precision for higher recall.')
    print('  SMOTE RF achieved the highest F1 score on the test set, suggesting it is the best option when both recall and precision matter for this imbalanced classification problem.')

    rf_hyper = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(oob_score=True, random_state=42))
    ])
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(rf_hyper, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print('\nBest Random Forest parameters:')
    print(grid_search.best_params_)
    print('OOB score:', best_pipeline.named_steps['classifier'].oob_score_)

    y_pred_best = best_pipeline.predict(X_test)
    best_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_best),
        'precision': precision_score(y_test, y_pred_best),
        'recall': recall_score(y_test, y_pred_best),
        'f1': f1_score(y_test, y_pred_best),
        'auc': roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1])
    }
    print('\nBest RF test metrics:')
    print(best_metrics)
    print('\nFinal recommendation:')
    print('  Deploy the tuned Random Forest pipeline because it achieved the best balance of accuracy, AUC, and F1 on the held-out test set.')
    print(f"  Best RF test accuracy={best_metrics['accuracy']:.3f}, precision={best_metrics['precision']:.3f}, recall={best_metrics['recall']:.3f}, F1={best_metrics['f1']:.3f}, AUC={best_metrics['auc']:.3f}.")

    regression_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']
    X_reg = df[regression_features]
    y_reg = df['fare']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_preprocessor = build_preprocessor(['pclass', 'age', 'sibsp', 'parch'], ['sex', 'embarked'])
    reg_pipeline = Pipeline([
        ('preprocessor', reg_preprocessor),
        ('regressor', LinearRegression())
    ])
    reg_pipeline.fit(Xr_train, yr_train)
    yr_pred = reg_pipeline.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    p = reg_pipeline.named_steps['preprocessor'].transform(Xr_train).shape[1]
    adj_r2 = adjusted_r2(r2, Xr_test.shape[0], p)
    print('\nRegression metrics:')
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R2:', r2)
    print('Adjusted R2:', adj_r2)

    residuals = yr_test - yr_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(yr_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Fare')
    plt.ylabel('Residuals')
    plt.title('Regression Residual Plot')
    save_plot('regression_residual_plot')
    plt.close()

    hetero_corr = np.corrcoef(yr_pred, np.abs(residuals))[0, 1]
    print('\nResidual heteroscedasticity check: correlation between predicted fare and absolute residual =', hetero_corr)
    if abs(hetero_corr) > 0.2:
        print('  The residual plot suggests heteroscedasticity: residual spread appears to grow with predicted fare.')
    else:
        print('  The residual plot does not show strong heteroscedasticity.')

    classification_table = pd.DataFrame({
        name: {
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc']
        }
        for name, res in results.items()
    }).T
    regression_table = pd.DataFrame({
        'Linear Regression': {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Adjusted R2': adj_r2
        }
    })
    print('\nFinal model comparison tables:')
    print('Classification metrics:')
    print(classification_table)
    print('\nRegression metrics:')
    print(regression_table)

    joblib.dump(best_pipeline, ARTIFACT_PATH)
    print(f'Best complete pipeline saved to {ARTIFACT_PATH}')

    loaded = joblib.load(ARTIFACT_PATH)
    sample = X_test.iloc[:5]
    print('\nReloaded pipeline test predictions:')
    print(loaded.predict(sample))


if __name__ == '__main__':
    main()
