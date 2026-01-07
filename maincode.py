# All Necessary Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations, cycle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

from scipy.stats import chi2_contingency
from math import sqrt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import scatter_matrix
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load dataset
csv_path = r'/content/18.csv'
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
df.head()

# Clean categorical columns
df['Sleep Disorder'] = df['Sleep Disorder'].replace(['None', ''], 'No Disorder').fillna('No Disorder')
df['Sleep Disorder'] = df['Sleep Disorder'].astype(str).str.strip().str.title()
df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
df['Occupation'] = df['Occupation'].astype(str).str.strip().str.title()
df['BMI Category'] = df['BMI Category'].astype(str).str.strip().str.title()

# Parse Blood Pressure column into Systolic & Diastolic
def parse_bp(x):
    try:
        s, d = str(x).split('/')
        return int(s), int(d)
    except:
        return (np.nan, np.nan)

bp_parsed = df['Blood Pressure'].apply(parse_bp)
df['SystolicBP'] = bp_parsed.apply(lambda t: t[0])
df['DiastolicBP'] = bp_parsed.apply(lambda t: t[1])

# Drop the original Blood Pressure column
df.drop(columns=['Blood Pressure'], inplace=True)

# Convert numeric columns
numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                'Stress Level', 'Heart Rate', 'Daily Steps', 'SystolicBP', 'DiastolicBP']

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Sleep Efficiency can stay
df['Sleep Efficiency'] = df['Sleep Duration'] / 8.0

print("\nMissing values per column after cleaning:\n", df.isna().sum())
print("\nClass balance (Sleep Disorder):\n", df['Sleep Disorder'].value_counts())

# Outlier detection
def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out_mask = (series < lower) | (series > upper)
    return {'q1': q1, 'q3': q3, 'iqr': iqr, 'lower': lower, 'upper': upper, 'n_outliers': int(out_mask.sum())}

num_features = ['Sleep Duration', 'Heart Rate', 'SystolicBP', 'DiastolicBP', 'Daily Steps']
outlier_summary = {c: iqr_outliers(df[c].dropna()) for c in num_features}
print("\nOutlier Summary (IQR Method):\n", outlier_summary)

# Remove outliers
for feature in num_features:
    stats = iqr_outliers(df[feature].dropna())
    lower, upper = stats['lower'], stats['upper']
    df = df[(df[feature] >= lower) & (df[feature] <= upper)]

print("\nShape after outlier removal:", df.shape)

# Reset index and Person ID
df.reset_index(drop=True, inplace=True)

if 'Person ID' in df.columns:
    df['Person ID'] = range(1, len(df) + 1)
else:
    df.insert(0, 'Person ID', range(1, len(df) + 1))

print("\nPerson ID column has been reset and is now continuous from 1 to", len(df))

# Group statistics
stat_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
             'Stress Level', 'Heart Rate', 'Daily Steps', 'SystolicBP', 'DiastolicBP']

group_stats = df.groupby('Sleep Disorder')[stat_cols].agg(['mean', 'median', 'std', 'count']).round(3)
mode_q_sleep = df.groupby('Sleep Disorder')['Quality of Sleep'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename('Mode_Quality_of_Sleep')

print("\nGroup Statistics (Mean, Median, Std, Count):")
print(group_stats)
print("\nMode of Quality of Sleep by Disorder:")
print(mode_q_sleep)

# Visualization
plt.figure(figsize=(8, 5))
counts = df['Sleep Disorder'].value_counts()
plt.bar(counts.index.astype(str), counts.values)
plt.title('Distribution of Sleep Disorders')
plt.ylabel('Count')
plt.xlabel('Sleep Disorder')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

plt.figure(figsize=(8, 6))
disorders = df['Sleep Disorder'].unique()
data_to_plot = [df.loc[df['Sleep Disorder'] == d, 'Sleep Duration'].dropna() for d in disorders]
plt.boxplot(data_to_plot, labels=disorders, showmeans=True)
plt.title('Sleep Duration by Disorder Type')
plt.ylabel('Sleep Duration (hours)')
plt.show()

plt.figure(figsize=(12, 10))
features = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
scatter_matrix(df[features].dropna(), figsize=(12, 12), diagonal='hist')
plt.suptitle('Pairwise Scatter Matrix (Selected Features)', y=0.9)
plt.show()

# Save cleaned data
cleaned_csv = r"/content/sleep_disorder_cleaned2.csv"
df.to_csv(cleaned_csv, index=False)
print("\nCleaned dataset saved to:", cleaned_csv)

path = "/content/sleep_disorder_cleaned2.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
df.head()
target_col = "Sleep Disorder"

#drop Person ID as it is non-informative
df = df.drop(columns=["Person ID"], errors="ignore")

# encoding categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


if target_col in df.columns:
    t = df.pop(target_col)
    df.insert(0, target_col, t)

# compute correlation matrix
corr_matrix = df.corr(method='pearson')

#reorder rows/columns by absolute correlation with the target so plot and ranking match
order = corr_matrix[target_col].abs().sort_values(ascending=False).index.tolist()
corr_matrix = corr_matrix.loc[order, order]

# print correlation
corr_with_target = corr_matrix[target_col].sort_values(key=lambda s: s.abs(), ascending=False)
print("\nCorrelation of each feature with Sleep Disorder (ordered by strength):")
print(corr_with_target)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.set(style="white")
plt.grid(False)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    linewidths=0.3,
    linecolor='white',
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=35, ha="right")   # angled labels for readability
plt.yticks(rotation=0)
plt.title("Ordered Pearson Correlation Matrix (Sleep Disorder as Target)", fontsize=15, pad=12)
plt.subplots_adjust(bottom=0.28)
plt.tight_layout()
plt.show()

#rank and pick top5
top5_features = corr_with_target.drop(index=target_col).abs().sort_values(ascending=False).head(5)
print("\nTop 5 features most correlated with Sleep Disorder:")
print(top5_features)
categorical_cols = ["Gender", "Occupation", "BMI Category", "Age"]

results = []

def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    return sqrt(phi2corr/denom) if denom > 0 else np.nan

for col in categorical_cols:
    if col in df.columns:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, _ = chi2_contingency(contingency)
        cv = cramers_v(df[col], df[target_col])
        results.append({"Feature": col, "Chi2": chi2, "p-value": p, "Cramer's V": cv})

chi_df = pd.DataFrame(results)
print("\nChi-square test results (categorical features):")
print(chi_df)

#highlight significant results
significant = chi_df[chi_df["p-value"] < 0.05]
print("\nStatistically significant associations (p < 0.05):")
print(significant)
path = "/content/sleep_disorder_cleaned2.csv"
corr_csv = "/content/correlation_with_sleep_disorder.csv"
target = "Sleep Disorder"

df_orig = pd.read_csv(path)

#fetch top 5
try:
    top5_list = list(top5)  # use top5 from session if available
except NameError:
    if os.path.exists(corr_csv):
        s = pd.read_csv(corr_csv, index_col=0, squeeze=True, header=None)
        try: s = s.iloc[:, 0]
        except Exception: pass
        s = s[~s.index.str.contains(target)]
        top5_list = list(s.abs().sort_values(ascending=False).head(5).index)
    else:
        tmp = df_orig.copy()
        for c in tmp.select_dtypes(include=['object']).columns: tmp[c] = LabelEncoder().fit_transform(tmp[c].astype(str))
        s = tmp.corr(method='pearson')[target].drop(index=target).abs()
        top5_list = list(s.sort_values(ascending=False).head(5).index)

print("Top5 for visualization:", top5_list)

#boxplots
numeric_top5 = [c for c in top5_list if pd.api.types.is_numeric_dtype(df_orig[c])]
if numeric_top5:
    n = len(numeric_top5); cols = 3; rows = (n + cols - 1) // cols
    plt.figure(figsize=(5*cols, 4*rows))
    for i, f in enumerate(numeric_top5, 1):
        ax = plt.subplot(rows, cols, i)
        sns.boxplot(x=target, y=f, data=df_orig, palette="coolwarm", ax=ax, showfliers=False)
        ax.set_title(f"{f} vs {target}")
        # annotate medians
        medians = df_orig.groupby(target)[f].median()
        ticks = ax.get_xticks()
        for xt, (lbl, m) in zip(ticks, medians.items()):
            ax.text(xt, m, f"{m:.2f}", ha='center', va='bottom', color='black', fontsize=9, weight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric top5 features found for boxplots:", top5_list)

#categorical countplots
cat_candidates = ["Gender", "Occupation", "BMI Category", "Age Bracket"]
cat_cols = [c for c in cat_candidates if c in df_orig.columns]

for c in cat_cols:
    plt.figure(figsize=(7,4))
    ax = sns.countplot(x=c, hue=target, data=df_orig, palette="Set2")
    plt.title(f"{c} vs {target}")
    plt.xticks(rotation=45)
    # bars annotation
    totals = df_orig.groupby(c)[target].count()
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        pct = 100 * height / totals.iloc[int(p.get_x() + 0.5)]
        ax.text(x, height + 1, f"{int(pct)}%", ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()
path = "/content/sleep_disorder_cleaned2.csv"
corr_csv = "/content/correlation_with_sleep_disorder.csv"
target_col = "Sleep Disorder"

#load original data
df_orig = pd.read_csv(path)

#load top 5
try:
    top5_list = list(top5)
except NameError:
    if os.path.exists(corr_csv):
        s = pd.read_csv(corr_csv, index_col=0, header=None, squeeze=True)
        try: s = s.iloc[:,0]
        except Exception: pass
        s = s[~s.index.str.contains(target_col)]
        top5_list = list(s.abs().sort_values(ascending=False).head(5).index)
    else:
        tmp = df_orig.copy()
        for c in tmp.select_dtypes(include=['object']).columns:
            tmp[c] = LabelEncoder().fit_transform(tmp[c].astype(str))
        s = tmp.corr(method='pearson')[target_col].drop(index=target_col).abs()
        top5_list = list(s.sort_values(ascending=False).head(5).index)

print("Top5 used for combined analysis:", top5_list)

# prepare df for modeling: keep target as label-encoded y, but retain original labels for reporting
df = df_orig.copy()
df = df.drop(columns=["Person ID"], errors="ignore")

# encode target for modeling
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col].astype(str))
target_mapping = dict(enumerate(le_target.classes_))
print("Target mapping:", target_mapping)

# build interaction features between top5 numeric features
# i've created pairwise products and a simple ratio (a/b) if both numeric and non-zero median.
interaction_cols = []
for a, b in combinations(top5_list, 2):
    if a in df.columns and b in df.columns:
        if pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b]):
            col_prod = f"{a}__x__{b}"
            df[col_prod] = df[a] * df[b]
            interaction_cols.append(col_prod)
            # ratio (avoid division by zero)
            denom = df[b].median() if df[b].median() != 0 else 1e-6
            col_ratio = f"{a}__div__{b}"
            df[col_ratio] = df[a] / (df[b].replace(0, denom))
            interaction_cols.append(col_ratio)

print("Created interaction columns:", interaction_cols[:10])

# features to consider are original top5 and interaction columns
candidate_features = [f for f in top5_list if f in df.columns] + interaction_cols
print("Candidate features count:", len(candidate_features))

# prepare X with proper encoding:
# identify categorical among candidates
cat_feats = [c for c in candidate_features if not pd.api.types.is_numeric_dtype(df[c])]
num_feats = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]

print("Numeric candidate features:", num_feats)
print("Categorical candidate features:", cat_feats) #onehot encoded

# build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_feats)
    ],
    remainder="drop"
)

# build final X which is preprocessed
X_pre = preprocessor.fit_transform(df[candidate_features])
# build feature names after transformation for VIF and feature importance plotting
ohe = preprocessor.named_transformers_.get("cat")
ohe_cols = []
if cat_feats and hasattr(ohe, "get_feature_names_out"):
    ohe_cols = list(ohe.get_feature_names_out(cat_feats))
feature_names = num_feats + ohe_cols

# VIF calculation
# convert to DataFrame for VIF calculaion
Xvif = pd.DataFrame(X_pre, columns=feature_names)
# Add small noise to avoid perfect collinearity numeric problems
Xvif += np.random.normal(0, 1e-8, Xvif.shape)

vif_data = pd.DataFrame({
    "feature": Xvif.columns,
    "VIF": [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])]
}).sort_values("VIF", ascending=False)
print("\nTop VIFs (multicollinearity check):")
print(vif_data.head(15))

# quick correlation of candidate features with target (for comparison)
# for correlation we use label-encoded target y and original numeric columns (interactions are numeric)
corr_with_target = {}
for f in feature_names:
    # if feature came from one-hot then compute point-biserial via Pearson with y
    series = Xvif[f]
    corr_with_target[f] = np.corrcoef(series, y)[0,1]
corr_df = pd.Series(corr_with_target).abs().sort_values(ascending=False)
print("\nTop correlations (abs) of candidate features with Sleep Disorder:")
print(corr_df.head(15))

# train a Random forest classifier (stratified CV) and evaluate
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_pre, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print(f"\nRandomForest (5-fold) F1_macro: mean={scores.mean():.3f}, std={scores.std():.3f}")

# fit on train/test split for permutation importance
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.25, stratify=y, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Test set metrics:",
      f"Acc={accuracy_score(y_test,y_pred):.3f}",
      f"Prec={precision_score(y_test,y_pred, average='macro'):.3f}",
      f"Rec={recall_score(y_test,y_pred, average='macro'):.3f}",
      f"F1={f1_score(y_test,y_pred, average='macro'):.3f}")

# permutation importance (on test set) ---
perm_res = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
perm_importances = pd.Series(perm_res.importances_mean, index=feature_names).sort_values(ascending=False)
print("\nTop permutation importances:")
print(perm_importances.head(15))

# bootstrap stability of permutation importances
n_boot = 30
boot_imps = np.zeros((n_boot, len(feature_names)))
rng = np.random.RandomState(42)
for i in range(n_boot):
    # sample with replacement
    idx = rng.choice(len(X_pre), size=len(X_pre), replace=True)
    Xb, yb = X_pre[idx], y[idx]
    rf_b = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_b.fit(Xb, yb)
    res_b = permutation_importance(rf_b, Xb, yb, n_repeats=8, random_state=i, n_jobs=-1)
    boot_imps[i, :] = res_b.importances_mean

boot_mean = boot_imps.mean(axis=0)
boot_std = boot_imps.std(axis=0)
boot_df = pd.DataFrame({"feature": feature_names, "perm_mean": boot_mean, "perm_std": boot_std})
boot_df = boot_df.sort_values("perm_mean", ascending=False).reset_index(drop=True)
print("\nBootstrap permutation importance (top 10):")
print(boot_df.head(10))

# plots: correlation bar, VIF, permutation importance
plt.figure(figsize=(8, max(4, 0.25*len(corr_df))))
sns.barplot(x=corr_df.values[:20], y=corr_df.index[:20], palette="magma")
plt.title("Absolute correlation with Sleep Disorder (candidates)")
plt.xlabel("abs(Pearson r)")
plt.tight_layout()
plt.savefig("/content/combined_corr_candidates.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, max(4, 0.25*min(20, len(vif_data)))))
sns.barplot(x="VIF", y="feature", data=vif_data.head(20), palette="viridis")
plt.title("Top features by VIF (multicollinearity check)")
plt.tight_layout()
plt.savefig("/content/vif_candidates.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, max(4, 0.25*len(perm_importances))))
sns.barplot(x=perm_importances.values[:20], y=perm_importances.index[:20], palette="coolwarm")
plt.title("Permutation Importance (test set)")
plt.xlabel("Mean decrease in score (importance)")
plt.tight_layout()
plt.savefig("/content/perm_importance_test.png", bbox_inches="tight")
plt.show()

# save output
vif_data.to_csv("/content/vif_candidates.csv", index=False)
perm_importances.to_csv("/content/perm_importance_test.csv")
boot_df.to_csv("/content/perm_importance_bootstrap.csv", index=False)
print("\nSaved: vif_candidates.csv, perm_importance_test.csv, perm_importance_bootstrap.csv, and plots.")

# Load Data
path = "/content/sleep_disorder_cleaned2.csv"
df = pd.read_csv(path)

# Dropping unnecessary columns
df = df.drop(columns=["Person ID"], errors="ignore")
df

# defining target column
target_col = "Sleep Disorder"
X = df.drop(columns=[target_col])   #all columns except target
y = df[target_col]
# One-hot encode target
y_encoded = pd.get_dummies(y)

# Check the result
print(y_encoded.head())
print("Columns (classes):", y_encoded.columns.tolist())
# defining subsets of features
feature_sets = {
    "all_features": X.columns.tolist(),  # all features
    "top5_correlated_features": top5_list,     # top 5 most correlated features
    "top3_statistical": ["Gender", "Occupation", "BMI Category"],  # top 3 statistically significant features
    "lifestyle_only": ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "Daily Steps"],  # lifestyle-related features
    "physical_only": ["BMI Category", "SystolicBP", "DiastolicBP", "Heart Rate", "Age", "Sleep Efficiency"] # physical features
}

# Checking the subsets
for k,v in feature_sets.items():
    print(f"{k}: ({len(v)} features): {v}")

results = {}  # Accuracy per subset
detailed_reports = {}
top_features_dict = {}

print("\nRandom Forest Evaluation Across Feature Subsets:\n")

for subset_name, features in feature_sets.items():
    X_subset = X[features]
    y_subset = y

    # Identify categorical features
    cat_feats = X_subset.select_dtypes(include=['object']).columns.tolist()
    num_feats = [col for col in features if col not in cat_feats]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_feats),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_feats)
        ]
    )

    model = Pipeline([
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=0.3, random_state=42, stratify=y_subset
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[subset_name] = acc

    # Print each heading with accuracy
    print(f"\nSubset: {subset_name} | Accuracy: {acc:.4f}\n")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    detailed_reports[subset_name] = report
    print(classification_report(y_test, y_pred))

    # Feature importances
    clf = model.named_steps['clf']
    feature_names = num_feats.copy()
    if cat_feats:
        cat_names = model.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out(cat_feats)
        feature_names += list(cat_names)
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_features_dict[subset_name] = importances
    print("Top 5 Features:")
    print(importances.head(5))
    # Summary table by Accuracy, plot and saving result
if results:
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)

    results_df = results_df.sort_values('Accuracy', ascending=False)
    print("Summary of Accuracies Across All Subsets")
    display(results_df)

    best_subset = results_df.sort_values(by="Accuracy", ascending=False).head(1)
    print("\nBest Performing Feature Subset:")
    print(best_subset)

    plt.figure(figsize=(10, max(4, len(results_df)*0.5)))
    sns.barplot(x='Accuracy', y=results_df.index, data=results_df, palette='Blues_d')
    plt.xlim(0, 1)
    plt.xlabel("Accuracy")
    plt.ylabel("")
    plt.title("Random Forest Accuracy by Feature Subset")
    plt.grid(axis = 'x')
    plt.tight_layout()
    plt.show()

    results_df.to_csv("/content/rf_model_accuracies_by_subset.csv")

    # Load dataset
df = pd.read_csv("sleep_disorder_cleaned2.csv")

# Use top 5 correlated features as identified by previous members
features = ["Physical Activity Level", "Daily Steps", "DiastolicBP", "SystolicBP", "Age"]
target = "Sleep Disorder"
X = df[features].copy()
y = df[target].copy()

# Identify categorical features (same approach as member 3)
cat_feats = X.select_dtypes(include=['object']).columns.tolist()
num_feats = [col for col in features if col not in cat_feats]

# Create preprocessing pipeline (same as member 3)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_feats),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_feats)
    ]
)

# Apply preprocessing to create processed datasets
X_processed = preprocessor.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=10,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print("Best Hyperparameters:")
print(grid_search.best_params_)
# Cross-validation evaluation on training set
cv_accuracy = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
cv_precision = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='precision_macro')
cv_recall = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='recall_macro')
cv_f1 = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1_macro')

print("\nCross-Validation Results (5-Fold on Training Set):")
print(f"CV Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"CV Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})")
print(f"CV Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
print(f"CV F1-score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

# Test set evaluation
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# Get unique classes from target variable
classes = np.unique(y)

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Binarize the output labels
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_rf.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(7, 6))
colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green"])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Multi-class ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
# Feature Importances
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances - Random Forest')
plt.show()

print("\nFeature Importances:")
print(feature_importance_df)
