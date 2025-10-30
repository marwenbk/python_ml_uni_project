# Student Performance Analysis - EDA and Feature Selection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.feature_selection import (
    VarianceThreshold,
    RFE,
    f_regression,
    mutual_info_regression,
)
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============ Load Data ============
df = pd.read_csv("student-por.csv")
print(df.shape)
df.head()

# ============ Basic Info ============
print("\n=== Dataset Info ===")
print(df.info())
print("\n=== Summary Statistics ===")
df.describe().T

print("\n=== Missing Values ===")
df.isnull().sum()

print("\n=== Duplicates ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ============ EDA - Categorical Variables ============

# Add readable labels for visualization
edu_map = {
    0: "None",
    1: "Primary (4th grade)",
    2: "Secondary (5th to 9th grade)",
    3: "High School",
    4: "Higher Education",
}
traveltime_map = {1: "<15 min", 2: "15–30 min", 3: "30 min–1 hour", 4: ">1 hour"}
studytime_map = {1: "<2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: ">10 hours"}
famrel_map = {1: "Very Bad", 2: "Bad", 3: "Good", 4: "Very Good", 5: "Excellent"}
scale5_map = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
health_map = {1: "Very Bad", 2: "Bad", 3: "Good", 4: "Very Good", 5: "Excellent"}

df["Medu_lbl"] = df["Medu"].map(edu_map)
df["Fedu_lbl"] = df["Fedu"].map(edu_map)
df["traveltime_lbl"] = df["traveltime"].map(traveltime_map)
df["studytime_lbl"] = df["studytime"].map(studytime_map)
df["famrel_lbl"] = df["famrel"].map(famrel_map)
df["freetime_lbl"] = df["freetime"].map(scale5_map)
df["goout_lbl"] = df["goout"].map(scale5_map)
df["Dalc_lbl"] = df["Dalc"].map(scale5_map)
df["Walc_lbl"] = df["Walc"].map(scale5_map)
df["health_lbl"] = df["health"].map(health_map)

# Categorical columns for plotting
cat_cols = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Medu_lbl",
    "Fedu_lbl",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime_lbl",
    "studytime_lbl",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel_lbl",
    "freetime_lbl",
    "goout_lbl",
    "Dalc_lbl",
    "Walc_lbl",
    "health_lbl",
]

# Count plots grid
sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"axes.titleweight": "bold"})

n_cat = len(cat_cols)
cols_grid = 3
rows_grid = math.ceil(n_cat / cols_grid)
fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(18, 5 * rows_grid))
axes = axes.flatten()
palette = sns.color_palette("rocket_r")

for i, col in enumerate(cat_cols):
    ax = axes[i]
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, ax=ax, order=order, palette=palette)
    ax.set_title(f"{col} distribution", fontsize=15)
    ax.tick_params(axis="x", rotation=45, labelsize=11)

    total = df[col].dropna().shape[0]
    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        x = p.get_x() + p.get_width() / 2
        y = height
        pct = height / total
        ax.text(
            x,
            y + max(1, total * 0.01),
            f"{int(height)}\n({pct:.0%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )

for j in range(n_cat, rows_grid * cols_grid):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Donut charts
pie_cols = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Medu_lbl",
    "Fedu_lbl",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime_lbl",
    "studytime_lbl",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

n_pies = len(pie_cols)
rows_pie = math.ceil(n_pies / 2)
fig, axes = plt.subplots(rows_pie, 2, figsize=(16, 5 * rows_pie))
axes = axes.flatten()
colors = sns.color_palette("Set2")

for i, col in enumerate(pie_cols):
    ax = axes[i]
    counts = df[col].value_counts()
    wedges, _ = ax.pie(
        counts.values,
        startangle=90,
        wedgeprops=dict(width=0.45),
        colors=colors[: len(counts)],
    )
    ax.set_title(f"Distribution of {col}", fontsize=14)
    ax.legend(
        wedges,
        counts.index,
        title=col,
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        fontsize=10,
    )
    ax.text(
        0,
        0,
        f"n={int(counts.sum())}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

if n_pies % 2 == 1:
    fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# ============ EDA - Numeric Variables ============

num_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3",
]

# Histograms
n_num = len(num_cols)
cols_num = 2
rows_num = math.ceil(n_num / cols_num)
fig, axes = plt.subplots(rows_num, cols_num, figsize=(16, 4 * rows_num))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    rng = df[col].dropna()
    bins = min(20, int(np.sqrt(len(rng))) + 5)
    sns.histplot(rng, bins=bins, kde=True, ax=ax, color=sns.color_palette("mako")[2])
    ax.set_title(f"{col} distribution", fontsize=13)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

for j in range(n_num, rows_num * cols_num):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Boxplots
fig, axes = plt.subplots(rows_num, cols_num, figsize=(14, 4 * rows_num))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, orient="h", palette="pastel")
    ax.set_title(f"{col} boxplot", fontsize=13)
    ax.set_xlabel(col)

for j in range(n_num, rows_num * cols_num):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# ============ Bivariate Analysis - Categorical vs G3 ============

print("\n=== Bivariate Analysis: Categorical Features vs G3 ===")

# Categorical features to analyze (using original encoded columns)
cat_features = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

# Box plots for categorical features vs G3
n_cat_bi = len(cat_features)
cols_cat_bi = 3
rows_cat_bi = math.ceil(n_cat_bi / cols_cat_bi)
fig, axes = plt.subplots(rows_cat_bi, cols_cat_bi, figsize=(18, 5 * rows_cat_bi))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ax = axes[i]
    sns.boxplot(data=df, x=col, y="G3", ax=ax, palette="Set2")
    ax.set_title(f"G3 distribution by {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Final Grade (G3)")

for j in range(n_cat_bi, rows_cat_bi * cols_cat_bi):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Mean G3 by category with error bars
fig, axes = plt.subplots(rows_cat_bi, cols_cat_bi, figsize=(18, 5 * rows_cat_bi))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ax = axes[i]
    grouped = df.groupby(col)["G3"].agg(["mean", "std", "count"])
    x_pos = range(len(grouped))
    ax.bar(
        x_pos,
        grouped["mean"],
        yerr=grouped["std"],
        capsize=5,
        alpha=0.7,
        color=sns.color_palette("viridis", len(grouped)),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45)
    ax.set_title(f"Mean G3 by {col}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean G3")
    ax.set_xlabel(col)

    # Add value labels on bars
    for j, (idx, row) in enumerate(grouped.iterrows()):
        ax.text(
            j,
            row["mean"] + row["std"] + 0.3,
            f"{row['mean']:.1f}\n(n={int(row['count'])})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

for j in range(n_cat_bi, rows_cat_bi * cols_cat_bi):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Statistical summary table
print("\n=== G3 Statistics by Categorical Features ===")
for col in cat_features:
    print(f"\n{col}:")
    summary = df.groupby(col)["G3"].agg(
        ["mean", "median", "std", "min", "max", "count"]
    )
    print(summary.round(2))

# ============ Bivariate Analysis - Numeric vs G3 ============

print("\n=== Bivariate Analysis: Numeric Features vs G3 ===")

# Numeric features to analyze (excluding G3 itself)
num_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

# Scatter plots with correlation
n_num_bi = len(num_features)
cols_num_bi = 3
rows_num_bi = math.ceil(n_num_bi / cols_num_bi)
fig, axes = plt.subplots(rows_num_bi, cols_num_bi, figsize=(18, 5 * rows_num_bi))
axes = axes.flatten()

for i, col in enumerate(num_features):
    ax = axes[i]
    # Scatter plot
    ax.scatter(df[col], df["G3"], alpha=0.5, s=30, color=sns.color_palette("mako")[3])

    # Add trend line
    z = np.polyfit(df[col].dropna(), df["G3"][df[col].notna()], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Calculate correlation
    corr = df[[col, "G3"]].corr().iloc[0, 1]

    ax.set_title(f"{col} vs G3 (r={corr:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Final Grade (G3)")
    ax.grid(True, alpha=0.3)

for j in range(n_num_bi, rows_num_bi * cols_num_bi):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Correlation coefficients table
print("\n=== Correlation with G3 ===")
correlations = (
    df[num_features + ["G3"]].corr()["G3"].drop("G3").sort_values(ascending=False)
)
print(correlations)

# Visualize correlations as bar chart
plt.figure(figsize=(10, 6))
correlations.plot(kind="barh", color=sns.color_palette("coolwarm", len(correlations)))
plt.title("Feature Correlations with G3", fontsize=15, fontweight="bold")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.show()

# ============ Grade Progression Analysis (G1 → G2 → G3) ============

print("\n=== Grade Progression Analysis ===")

# Side-by-side distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
grade_cols = ["G1", "G2", "G3"]
colors = ["skyblue", "lightcoral", "lightgreen"]

for i, (grade, color) in enumerate(zip(grade_cols, colors)):
    ax = axes[i]
    sns.histplot(df[grade], bins=20, kde=True, ax=ax, color=color)
    ax.set_title(f"{grade} Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel(grade)
    ax.set_ylabel("Frequency")
    mean_val = df[grade].mean()
    median_val = df[grade].median()
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.2f}",
    )
    ax.legend()

plt.tight_layout()
plt.show()

# Correlation matrix for grades
print("\n=== Grade Correlation Matrix ===")
grade_corr = df[["G1", "G2", "G3"]].corr()
print(grade_corr.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(
    grade_corr,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    center=0.5,
    square=True,
    linewidths=2,
    cbar_kws={"label": "Correlation"},
)
plt.title("Grade Progression Correlation Matrix", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# Grade transition scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
grade_pairs = [("G1", "G2"), ("G2", "G3"), ("G1", "G3")]

for i, (g1, g2) in enumerate(grade_pairs):
    ax = axes[i]
    ax.scatter(
        df[g1], df[g2], alpha=0.5, s=30, color=sns.color_palette("viridis")[i * 2]
    )

    # Add trend line
    z = np.polyfit(df[g1].dropna(), df[g2][df[g1].notna()], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[g1].min(), df[g1].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Add diagonal reference line (perfect correlation)
    min_val = min(df[g1].min(), df[g2].min())
    max_val = max(df[g1].max(), df[g2].max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k:",
        alpha=0.5,
        linewidth=1.5,
        label="y=x",
    )

    corr = df[[g1, g2]].corr().iloc[0, 1]
    ax.set_title(f"{g1} vs {g2} (r={corr:.3f})", fontsize=14, fontweight="bold")
    ax.set_xlabel(g1)
    ax.set_ylabel(g2)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Grade progression statistics
print("\n=== Grade Progression Statistics ===")
print(df[["G1", "G2", "G3"]].describe().round(2))

# Calculate grade changes
df["G1_to_G2_change"] = df["G2"] - df["G1"]
df["G2_to_G3_change"] = df["G3"] - df["G2"]
df["G1_to_G3_change"] = df["G3"] - df["G1"]

print("\n=== Grade Change Statistics ===")
print(
    f"G1 to G2 change: mean={df['G1_to_G2_change'].mean():.2f}, std={df['G1_to_G2_change'].std():.2f}"
)
print(
    f"G2 to G3 change: mean={df['G2_to_G3_change'].mean():.2f}, std={df['G2_to_G3_change'].std():.2f}"
)
print(
    f"G1 to G3 change: mean={df['G1_to_G3_change'].mean():.2f}, std={df['G1_to_G3_change'].std():.2f}"
)

# Visualize grade changes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
change_cols = ["G1_to_G2_change", "G2_to_G3_change", "G1_to_G3_change"]
titles = ["G1 → G2 Change", "G2 → G3 Change", "G1 → G3 Change"]

for i, (col, title) in enumerate(zip(change_cols, titles)):
    ax = axes[i]
    sns.histplot(
        df[col], bins=30, kde=True, ax=ax, color=sns.color_palette("rocket")[i * 2]
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Grade Change")
    ax.set_ylabel("Frequency")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No Change")
    ax.legend()

plt.tight_layout()
plt.show()

# Drop temporary change columns
df = df.drop(columns=["G1_to_G2_change", "G2_to_G3_change", "G1_to_G3_change"])

# ============ Data Preparation ============

# Reload original data
df = pd.read_csv("student-por.csv")

# Binary mapping
binary_map = {
    "yes": 1,
    "no": 0,
    "M": 1,
    "F": 0,
    "GP": 1,
    "MS": 0,
    "U": 1,
    "R": 0,
    "GT3": 1,
    "LE3": 0,
    "T": 1,
    "A": 0,
}
df = df.replace(binary_map)

# One-hot encode categorical
encode_cols = ["Mjob", "Fjob", "reason", "guardian"]
df = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)

# Drop age
if "age" in df.columns:
    df = df.drop(columns=["age"])

# Split X and y
y = df["G3"]
X = df.drop(columns=["G3"])

# Remove zero variance
vt = VarianceThreshold(threshold=0.0)
X = X.loc[:, vt.fit(X).get_support()]

print(f"\nX shape: {X.shape}, y shape: {y.shape}")

# Identify binary vs multi-valued columns
uniq = X.nunique(dropna=False)
binary_cols = uniq[uniq == 2].index.tolist()
multi_cols = uniq[uniq > 2].index.tolist()
print(f"Binary columns: {len(binary_cols)}, Multi-valued columns: {len(multi_cols)}")

# ============ Correlation Heatmaps ============

if len(multi_cols) >= 2:
    # Pearson
    corr_p = X[multi_cols].corr(method="pearson")
    corr_p_pct = corr_p * 100.0
    annot_p = corr_p_pct.round(0).astype(int).astype(str) + "%"

    plt.figure(figsize=(max(8, 0.5 * len(multi_cols)), max(6, 0.5 * len(multi_cols))))
    sns.heatmap(
        corr_p_pct,
        annot=annot_p,
        fmt="",
        vmin=-100,
        vmax=100,
        cmap="coolwarm",
        cbar_kws={"label": "Pearson correlation (%)"},
    )
    plt.title("Pearson correlation (%, multi-valued columns)")
    plt.tight_layout()
    plt.show()

    # Spearman
    corr_s = X[multi_cols].corr(method="spearman")
    corr_s_pct = corr_s * 100.0
    annot_s = corr_s_pct.round(0).astype(int).astype(str) + "%"

    plt.figure(figsize=(max(8, 0.5 * len(multi_cols)), max(6, 0.5 * len(multi_cols))))
    sns.heatmap(
        corr_s_pct,
        annot=annot_s,
        fmt="",
        vmin=-100,
        vmax=100,
        cmap="coolwarm",
        cbar_kws={"label": "Spearman correlation (%)"},
    )
    plt.title("Spearman correlation (%, multi-valued columns)")
    plt.tight_layout()
    plt.show()

# ============ Multicollinearity Detection ============

print("\n=== Multicollinearity Detection ===")

# Calculate VIF for all features
print("\n=== Calculating Variance Inflation Factor (VIF) ===")
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]
vif_data = vif_data.sort_values("VIF", ascending=False)

print("\n=== VIF Analysis ===")
print(vif_data)

# Highlight problematic features
vif_severe = vif_data[vif_data["VIF"] > 10]
vif_moderate = vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)]

print(f"\n=== Severe Multicollinearity (VIF > 10): {len(vif_severe)} features ===")
if len(vif_severe) > 0:
    print(vif_severe)

print(
    f"\n=== Moderate Multicollinearity (5 < VIF <= 10): {len(vif_moderate)} features ==="
)
if len(vif_moderate) > 0:
    print(vif_moderate)

# Visualize VIF
plt.figure(figsize=(12, max(6, len(X.columns) * 0.3)))
colors = ["red" if v > 10 else "orange" if v > 5 else "green" for v in vif_data["VIF"]]
plt.barh(range(len(vif_data)), vif_data["VIF"], color=colors)
plt.yticks(range(len(vif_data)), vif_data["Feature"])
plt.xlabel("VIF Value", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title(
    "Variance Inflation Factor (VIF) for All Features", fontsize=14, fontweight="bold"
)
plt.axvline(
    x=5, color="orange", linestyle="--", linewidth=2, label="VIF = 5 (Moderate)"
)
plt.axvline(x=10, color="red", linestyle="--", linewidth=2, label="VIF = 10 (Severe)")
plt.legend()
plt.tight_layout()
plt.show()

# Identify highly correlated feature pairs (correlation > 0.9)
print("\n=== Highly Correlated Feature Pairs (|correlation| > 0.9) ===")
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_pairs = []

for column in upper_triangle.columns:
    high_corr_features = upper_triangle.index[upper_triangle[column] > 0.9].tolist()
    for feature in high_corr_features:
        corr_val = corr_matrix.loc[feature, column]
        # Calculate correlation with target
        corr_with_target_1 = abs(X[feature].corr(y))
        corr_with_target_2 = abs(X[column].corr(y))
        high_corr_pairs.append(
            {
                "Feature_1": feature,
                "Feature_2": column,
                "Correlation": corr_val,
                "Feature_1_to_Target": corr_with_target_1,
                "Feature_2_to_Target": corr_with_target_2,
            }
        )

if len(high_corr_pairs) > 0:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df)
else:
    print("No feature pairs with correlation > 0.9 found.")

# ============ Multicollinearity Resolution ============

print("\n=== Multicollinearity Resolution ===")
print(f"Original feature count: {X.shape[1]}")

# Create a copy for resolution
X_clean = X.copy()

# Strategy 1: Remove features with VIF > 10
features_to_remove_vif = set()
if len(vif_severe) > 0:
    print("\n=== Removing features with VIF > 10 ===")
    features_to_remove_vif = set(vif_severe["Feature"].tolist())
    print(f"Features to remove based on VIF: {features_to_remove_vif}")

# Strategy 2: Remove one feature from highly correlated pairs (>0.9)
features_to_remove_corr = set()
if len(high_corr_pairs) > 0:
    print("\n=== Resolving highly correlated pairs (correlation > 0.9) ===")
    for pair in high_corr_pairs:
        feat1, feat2 = pair["Feature_1"], pair["Feature_2"]
        corr1, corr2 = pair["Feature_1_to_Target"], pair["Feature_2_to_Target"]

        # Keep the feature with higher correlation to target
        if corr1 >= corr2:
            features_to_remove_corr.add(feat2)
            print(
                f"Pair ({feat1}, {feat2}): Keeping {feat1} (|r|={corr1:.3f}), Removing {feat2} (|r|={corr2:.3f})"
            )
        else:
            features_to_remove_corr.add(feat1)
            print(
                f"Pair ({feat1}, {feat2}): Keeping {feat2} (|r|={corr2:.3f}), Removing {feat1} (|r|={corr1:.3f})"
            )

# Combine features to remove
features_to_remove = features_to_remove_vif.union(features_to_remove_corr)

# Remove features
if len(features_to_remove) > 0:
    print(f"\n=== Total features to remove: {len(features_to_remove)} ===")
    print(f"Features: {sorted(features_to_remove)}")
    X_clean = X_clean.drop(columns=list(features_to_remove))
    print(f"\nFeatures after multicollinearity resolution: {X_clean.shape[1]}")

    # Recalculate VIF for cleaned data
    print("\n=== Recalculating VIF after cleanup ===")
    vif_data_clean = pd.DataFrame()
    vif_data_clean["Feature"] = X_clean.columns
    vif_data_clean["VIF"] = [
        variance_inflation_factor(X_clean.values, i)
        for i in range(len(X_clean.columns))
    ]
    vif_data_clean = vif_data_clean.sort_values("VIF", ascending=False)

    print("\n=== VIF After Cleanup (Top 20) ===")
    print(vif_data_clean.head(20))

    # Visualize VIF comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(X_clean.columns) * 0.2)))

    # Before
    ax1 = axes[0]
    colors_before = [
        "red" if v > 10 else "orange" if v > 5 else "green" for v in vif_data["VIF"]
    ]
    ax1.barh(range(len(vif_data)), vif_data["VIF"], color=colors_before)
    ax1.set_yticks(range(len(vif_data)))
    ax1.set_yticklabels(vif_data["Feature"], fontsize=8)
    ax1.set_xlabel("VIF Value")
    ax1.set_title(f"VIF Before Cleanup ({len(X.columns)} features)", fontweight="bold")
    ax1.axvline(x=10, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # After
    ax2 = axes[1]
    colors_after = [
        "red" if v > 10 else "orange" if v > 5 else "green"
        for v in vif_data_clean["VIF"]
    ]
    ax2.barh(range(len(vif_data_clean)), vif_data_clean["VIF"], color=colors_after)
    ax2.set_yticks(range(len(vif_data_clean)))
    ax2.set_yticklabels(vif_data_clean["Feature"], fontsize=8)
    ax2.set_xlabel("VIF Value")
    ax2.set_title(
        f"VIF After Cleanup ({len(X_clean.columns)} features)", fontweight="bold"
    )
    ax2.axvline(x=10, color="red", linestyle="--", linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Update X to use cleaned data
    X = X_clean
    print(f"\n=== Updated X with cleaned features: {X.shape} ===")

    # Update binary and multi-valued columns
    uniq = X.nunique(dropna=False)
    binary_cols = uniq[uniq == 2].index.tolist()
    multi_cols = uniq[uniq > 2].index.tolist()
    print(
        f"Binary columns: {len(binary_cols)}, Multi-valued columns: {len(multi_cols)}"
    )
else:
    print(
        "\nNo features need to be removed. Multicollinearity is within acceptable limits."
    )

# ============ Feature Selection ============

# F-regression
f_vals, p_vals = f_regression(X, y)
freg_df = pd.DataFrame({"feature": X.columns, "f_score": f_vals, "pval": p_vals})
freg_df = freg_df.sort_values("f_score", ascending=False)

print("\n=== Top 20 Features by F-Regression ===")
print(freg_df.head(20))

plt.figure(figsize=(10, 4))
freg_df.head(10).set_index("feature")["f_score"].plot(kind="bar")
plt.ylabel("F-score (f_regression)")
plt.title("Top features by f_regression")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({"feature": X.columns, "mutual_info": mi_scores})
mi_df = mi_df.sort_values("mutual_info", ascending=False)

print("\n=== Top 20 Features by Mutual Information ===")
print(mi_df.head(20))

plt.figure(figsize=(10, 4))
mi_df.head(10).set_index("feature")["mutual_info"].plot(kind="bar")
plt.ylabel("Mutual information")
plt.title("Top features by mutual_info_regression")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# RFE with Random Forest
n_rfe = 10 if X.shape[1] >= 10 else max(1, X.shape[1])
rfe_est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_est, n_features_to_select=n_rfe, step=1)
rfe.fit(X, y)

rfe_df = pd.DataFrame({"feature": X.columns, "rfe_rank": rfe.ranking_})
rfe_df["rfe_score"] = 1.0 / rfe_df["rfe_rank"]
rfe_df = rfe_df.sort_values("rfe_rank")

print("\n=== Top RFE Features ===")
print(rfe_df.head(20))

plt.figure(figsize=(10, 4))
rfe_df.head(10).set_index("feature")["rfe_score"].plot(kind="bar")
plt.ylabel("RFE score (1/rank)")
plt.title("Top RFE features (RandomForestRegressor)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Combine top features
k_select = 10
selected_f = set(freg_df.head(k_select)["feature"])
selected_mi = set(mi_df.head(k_select)["feature"])
selected_rfe = set(rfe_df.head(n_rfe)["feature"])
selected_union = list(selected_f | selected_mi | selected_rfe)

X_final = X[selected_union]

print("\n=== Final Selected Features ===")
print(f"Total selected: {len(selected_union)} features")
print(f"Features: {sorted(selected_union)}")
print(f"X_final shape: {X_final.shape}")
print("\nNote: For tree models (RF, XGBoost) - no scaling needed")
print("For linear models (Ridge, Lasso) or neural nets - use StandardScaler")
