# Titanic Analytics Pipeline - Part A
# Run this in VS Code (Python environment with seaborn, pandas, matplotlib, scikit-learn installed)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

output_dir = Path(__file__).resolve().parent
plot_dir = output_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


def save_plot(name):
    filename = plot_dir / f"{name}.png"
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved plot: {filename}")


# -------------------------------
# Task 1: Load dataset & profile
# -------------------------------
df = sns.load_dataset('titanic')
output_path = output_dir / "titanic.csv"
df.to_csv(output_path, index=False)   # offline fallback

print("=== Dataset Info ===")
df.info()
print("\n=== Dataset Shape ===", df.shape)
print("\n=== Dataset Description ===")
print(df.describe(include='all'))

# Missing values percentage
missing = df.isnull().mean() * 100
print("\n=== Missing Value Percentages ===")
print(missing[missing > 0])

# -------------------------------
# Task 2: Missing value handling
# -------------------------------
clean_df = df.copy()
print("\n=== Missing Value Handling Decisions ===")
for col in df.columns:
    miss_pct = df[col].isnull().mean() * 100
    if miss_pct == 0:
        continue
    if miss_pct < 5:
        clean_df = clean_df.dropna(subset=[col])
        print(f"{col}: {miss_pct:.2f}% missing → drop rows because missingness is low and row removal preserves data quality.")
    elif miss_pct <= 30:
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_df[col].fillna(df[col].median(), inplace=True)
            print(f"{col}: {miss_pct:.2f}% missing → impute with median because numeric imputation is reliable at this rate.")
        else:
            clean_df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"{col}: {miss_pct:.2f}% missing → impute with mode because categorical imputation is reasonable at this rate.")
    else:
        if col == 'deck':
            clean_df.drop(columns=[col], inplace=True)
            print(f"{col}: {miss_pct:.2f}% missing → drop column because missingness is too high for reliable imputation and the feature is not essential for the core analysis.")
        elif pd.api.types.is_numeric_dtype(df[col]):
            clean_df[col].fillna(df[col].median(), inplace=True)
            print(f"{col}: {miss_pct:.2f}% missing → impute numeric values with median despite high missingness, preserving the signal where possible.")
        else:
            clean_df[col] = clean_df[col].fillna("Missing")
            print(f"{col}: {miss_pct:.2f}% missing → encode 'Missing' as its own category because the column is categorical and dropping it would lose useful information.")

print("\nAfter Cleaning, Shape:", clean_df.shape)

# -------------------------------
# Task 3: Univariate analysis
# -------------------------------
for col in ['age','fare']:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(clean_df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.subplot(1,2,2)
    sns.boxplot(x=clean_df[col])
    plt.title(f"Boxplot of {col}")
    save_plot(f"univariate_{col}")
    plt.show()

    Q1, Q3 = clean_df[col].quantile([0.25,0.75])
    IQR = Q3 - Q1
    outliers = ((clean_df[col] < Q1 - 1.5*IQR) | (clean_df[col] > Q3 + 1.5*IQR)).sum()
    print(f"{col} outliers (IQR rule):", outliers)

# Skewness check for fare
fare_mean = clean_df['fare'].mean()
fare_median = clean_df['fare'].median()
fare_mode = clean_df['fare'].mode()[0]
print("\nFare Mean:", fare_mean, "Median:", fare_median, "Mode:", fare_mode)
if fare_mean > fare_median > fare_mode:
    print("Fare distribution is Right-Skewed")
elif fare_mean < fare_median < fare_mode:
    print("Fare distribution is Left-Skewed")
else:
    print("Fare distribution is Symmetric")

# -------------------------------
# Task 4: Bivariate analysis
# -------------------------------
print("\nSurvival Rate by Sex (boolean masking):")
for sex in ['male', 'female']:
    mask = clean_df['sex'] == sex
    rate = clean_df.loc[mask, 'survived'].mean()
    print(f"  {sex}: {rate:.3f}")

print("\nSurvival Rate by Pclass (boolean masking):")
for pclass in sorted(clean_df['pclass'].unique()):
    mask = clean_df['pclass'] == pclass
    rate = clean_df.loc[mask, 'survived'].mean()
    print(f"  Pclass {pclass}: {rate:.3f}")

print("\nSurvival Rate by Sex and Pclass (boolean masking):")
for sex in ['male', 'female']:
    for pclass in sorted(clean_df['pclass'].unique()):
        mask = (clean_df['sex'] == sex) & (clean_df['pclass'] == pclass)
        rate = clean_df.loc[mask, 'survived'].mean()
        print(f"  {sex}, Pclass {pclass}: {rate:.3f}")

# Correlation matrix (6 specified columns)
corr_cols = ['survived','pclass','age','sibsp','parch','fare']
corr_matrix = clean_df[corr_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (6 columns)")
save_plot("correlation_heatmap")
plt.show()

# Identify strongest correlations
abs_corr = corr_matrix.abs()
abs_corr = abs_corr.where(~np.eye(len(abs_corr), dtype=bool), 0)
strong_pairs = abs_corr.unstack().sort_values(ascending=False).drop_duplicates()
print("\nTop 2 strongest correlations:")
for (a, b), value in strong_pairs.head(2).items():
    corr_value = corr_matrix.loc[a, b]
    print(f"  {a} vs {b}: correlation = {corr_value:.3f} (abs={abs(value):.3f})")
print("\nInterpretation:")
if not strong_pairs.empty:
    (a1, b1), value1 = strong_pairs.head(1).items().__iter__().__next__()
    print(f"  The strongest relationship is between {a1} and {b1}, indicating that these two numeric features move together most strongly in the dataset.")
if len(strong_pairs) > 1:
    (a2, b2), value2 = strong_pairs.iloc[1:2].items().__iter__().__next__()
    print(f"  The second strongest relationship is between {a2} and {b2}, showing the next most important numeric association among the selected features.")

# -------------------------------
# Task 5: Multivariate data story
# -------------------------------
plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='survived', data=clean_df)
plt.title("Survival by Sex")
save_plot("survival_by_sex")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='pclass', y='survived', data=clean_df)
plt.title("Survival by Pclass")
save_plot("survival_by_pclass")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='survived', y='age', data=clean_df)
plt.title("Age vs Survival")
save_plot("age_vs_survival")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='fare', y='age', hue='survived', data=clean_df)
plt.title("Fare vs Age colored by Survival")
save_plot("fare_vs_age_survival")
plt.show()

# -------------------------------
# Task 6: Standardization check
# -------------------------------
scaler = StandardScaler()
clean_df[['age_scaled','fare_scaled']] = scaler.fit_transform(clean_df[['age','fare']])

print("\nBefore Standardization:")
print(clean_df[['age','fare']].describe().loc[['mean','std']])
print("\nAfter Standardization:")
print(clean_df[['age_scaled','fare_scaled']].describe().loc[['mean','std']])

plt.figure(figsize=(10,4))
sns.kdeplot(clean_df['age'], label='Age Original')
sns.kdeplot(clean_df['age_scaled'], label='Age Scaled')
plt.legend()
plt.title("Age Distribution Before/After Standardization")
save_plot("age_standardization")
plt.show()

plt.figure(figsize=(10,4))
sns.kdeplot(clean_df['fare'], label='Fare Original')
sns.kdeplot(clean_df['fare_scaled'], label='Fare Scaled')
plt.legend()
plt.title("Fare Distribution Before/After Standardization")
save_plot("fare_standardization")
plt.show()
