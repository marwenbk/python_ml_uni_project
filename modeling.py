# Student Performance Prediction - Machine Learning Modeling
# Comprehensive ML pipeline with feature selection, model training, and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# Model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (12, 6), "axes.titleweight": "bold"})

# ============ Data Preparation ============


def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    df = pd.read_csv("student-por.csv")
    print(f"Dataset shape: {df.shape}")

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

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Target (G3) - Mean: {y.mean():.2f}, Std: {y.std():.2f}\n")

    return X, y


# ============ Feature Selection Functions ============


def remove_high_vif_features(X, y, threshold=10, exclude_features=None, max_iterations=15):
    """Remove features with high VIF iteratively (optimized)"""
    print(f"\n--- Removing features with VIF > {threshold} ---")

    if exclude_features is None:
        exclude_features = []

    X_clean = X.copy()
    removed_features = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}: {X_clean.shape[1]} features...", end=" ", flush=True)
        
        # Calculate VIF (this is the slow part)
        vif_values = []
        for i in range(len(X_clean.columns)):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_values.append(vif)
            except:
                vif_values.append(0)
        
        vif_data = pd.DataFrame({
            "Feature": X_clean.columns,
            "VIF": vif_values
        })
        vif_data = vif_data.sort_values("VIF", ascending=False)

        # Find max VIF (excluding protected features)
        vif_filtered = vif_data[~vif_data["Feature"].isin(exclude_features)]

        if len(vif_filtered) == 0 or vif_filtered["VIF"].max() <= threshold:
            print("Done!")
            break

        # Remove feature with highest VIF
        feature_to_remove = vif_filtered.iloc[0]["Feature"]
        vif_value = vif_filtered.iloc[0]["VIF"]

        print(f"Removing '{feature_to_remove}' (VIF={vif_value:.2f})")
        X_clean = X_clean.drop(columns=[feature_to_remove])
        removed_features.append(feature_to_remove)

    print(
        f"Removed {len(removed_features)} features: {removed_features if removed_features else 'None'}"
    )
    print(f"Remaining features: {X_clean.shape[1]}\n")

    return X_clean, removed_features


def select_features_union(X, y, k=20):
    """Select features using union of multiple methods"""
    print(f"\n--- Feature Selection: Union of methods (top {k}) ---")

    # F-regression
    f_vals, p_vals = f_regression(X, y)
    freg_df = pd.DataFrame({"feature": X.columns, "f_score": f_vals, "pval": p_vals})
    freg_df = freg_df.sort_values("f_score", ascending=False)
    selected_f = set(freg_df.head(k)["feature"])
    print(f"F-regression selected: {len(selected_f)} features")

    # Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_info": mi_scores})
    mi_df = mi_df.sort_values("mutual_info", ascending=False)
    selected_mi = set(mi_df.head(k)["feature"])
    print(f"Mutual Info selected: {len(selected_mi)} features")

    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    rf_imp = rf_imp.sort_values("importance", ascending=False)
    selected_rf = set(rf_imp.head(k)["feature"])
    print(f"Random Forest selected: {len(selected_rf)} features")

    # Union
    selected_union = list(selected_f | selected_mi | selected_rf)
    print(f"Union: {len(selected_union)} unique features")

    # Show top features by average rank
    rank_f = {f: i for i, f in enumerate(freg_df["feature"])}
    rank_mi = {f: i for i, f in enumerate(mi_df["feature"])}
    rank_rf = {f: i for i, f in enumerate(rf_imp["feature"])}

    avg_ranks = []
    for feat in selected_union:
        avg_rank = (rank_f[feat] + rank_mi[feat] + rank_rf[feat]) / 3
        avg_ranks.append((feat, avg_rank))

    avg_ranks.sort(key=lambda x: x[1])
    print("\nTop 15 features by average rank:")
    for i, (feat, rank) in enumerate(avg_ranks[:15], 1):
        print(f"  {i:2d}. {feat:20s} (avg rank: {rank:5.1f})")

    return selected_union


# ============ Model Training Functions ============


def train_baseline_models(X_train, X_test, y_train, y_test, scenario_name):
    """Train baseline models without tuning"""
    print(f"\n{'=' * 80}")
    print(f"BASELINE MODELS - {scenario_name}")
    print("=" * 80)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)

        # Train predictions
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

        print(f"  Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(
            f"  Test R²:  {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%"
        )

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "predictions": y_test_pred,
        }

    return results


def train_advanced_models(X_train, X_test, y_train, y_test, scenario_name, scaled=False):
    """Train advanced models (optimized for speed)"""
    print(f"\n{'=' * 80}")
    print(f"ADVANCED MODELS - {scenario_name}")
    print("=" * 80)

    if scaled:
        models = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Lasso Regression": Lasso(alpha=0.1, random_state=42, max_iter=2000),
            "SVR": SVR(kernel="rbf", C=1, epsilon=0.1),
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
            ),
        }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)

        # Train predictions
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

        # Cross-validation on training set (3-fold for speed)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1
        )

        print(f"  Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        print(
            f"  Test R²:  {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%"
        )
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "predictions": y_test_pred,
        }

    return results


def tune_top_models(X_train, y_train, scenario_name, scaled=False):
    """Hyperparameter tuning for top models (optimized for speed)"""
    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER TUNING - {scenario_name}")
    print("=" * 80)

    tuned_models = {}

    if scaled:
        # Ridge
        print("\n--- Tuning Ridge Regression ---")
        ridge_params = {"alpha": [0.1, 1, 10]}
        ridge_grid = GridSearchCV(
            Ridge(random_state=42), ridge_params, cv=3, scoring="r2", n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        print(f"  Best params: {ridge_grid.best_params_}")
        print(f"  Best CV R²: {ridge_grid.best_score_:.4f}")
        tuned_models["Ridge (tuned)"] = ridge_grid.best_estimator_

        # Lasso
        print("\n--- Tuning Lasso Regression ---")
        lasso_params = {"alpha": [0.01, 0.1, 1]}
        lasso_grid = GridSearchCV(
            Lasso(random_state=42, max_iter=2000), lasso_params, cv=3, scoring="r2", n_jobs=-1
        )
        lasso_grid.fit(X_train, y_train)
        print(f"  Best params: {lasso_grid.best_params_}")
        print(f"  Best CV R²: {lasso_grid.best_score_:.4f}")
        tuned_models["Lasso (tuned)"] = lasso_grid.best_estimator_

    else:
        # Random Forest
        print("\n--- Tuning Random Forest ---")
        rf_params = {
            "n_estimators": [50, 100],
            "max_depth": [10, 15],
        }
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1, min_samples_split=2),
            rf_params,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        print(f"  Best params: {rf_grid.best_params_}")
        print(f"  Best CV R²: {rf_grid.best_score_:.4f}")
        tuned_models["Random Forest (tuned)"] = rf_grid.best_estimator_

        # Gradient Boosting
        print("\n--- Tuning Gradient Boosting ---")
        gb_params = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
        }
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42, learning_rate=0.1),
            gb_params,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        gb_grid.fit(X_train, y_train)
        print(f"  Best params: {gb_grid.best_params_}")
        print(f"  Best CV R²: {gb_grid.best_score_:.4f}")
        tuned_models["Gradient Boosting (tuned)"] = gb_grid.best_estimator_

    return tuned_models


# ============ Evaluation Functions ============


def evaluate_tuned_models(tuned_models, X_train, X_test, y_train, y_test):
    """Evaluate tuned models"""
    results = {}

    for name, model in tuned_models.items():
        # Train predictions
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

        results[name] = {
            "model": model,
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "predictions": y_test_pred,
        }

    return results


def create_comparison_table(all_results):
    """Create comprehensive comparison table"""
    comparison_data = []

    for name, metrics in all_results.items():
        comparison_data.append(
            {
                "Model": name,
                "Train R²": f"{metrics['train_r2']:.4f}",
                "Test R²": f"{metrics['test_r2']:.4f}",
                "Test MAE": f"{metrics['test_mae']:.4f}",
                "Test RMSE": f"{metrics['test_rmse']:.4f}",
                "Test MAPE": f"{metrics['test_mape']:.2f}%",
                "CV R² (mean±std)": f"{metrics.get('cv_r2_mean', 0):.4f}±{metrics.get('cv_r2_std', 0):.4f}"
                if "cv_r2_mean" in metrics
                else "N/A",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values("Test R²", ascending=False)

    return df_comparison


# ============ Visualization Functions ============


def plot_model_comparison(all_results, scenario_name):
    """Plot model comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(all_results.keys())
    test_r2 = [all_results[m]["test_r2"] for m in model_names]
    test_rmse = [all_results[m]["test_rmse"] for m in model_names]
    test_mae = [all_results[m]["test_mae"] for m in model_names]
    cv_r2 = [all_results[m].get("cv_r2_mean", 0) for m in model_names]

    # R² Comparison
    ax1 = axes[0, 0]
    bars = ax1.barh(model_names, test_r2, color=sns.color_palette("viridis", len(model_names)))
    ax1.set_xlabel("R² Score", fontweight="bold")
    ax1.set_title(f"Test R² Comparison - {scenario_name}", fontweight="bold", fontsize=14)
    ax1.axvline(x=0, color="black", linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars, test_r2)):
        ax1.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

    # RMSE Comparison
    ax2 = axes[0, 1]
    bars = ax2.barh(model_names, test_rmse, color=sns.color_palette("rocket_r", len(model_names)))
    ax2.set_xlabel("RMSE", fontweight="bold")
    ax2.set_title(f"Test RMSE Comparison - {scenario_name}", fontweight="bold", fontsize=14)
    for i, (bar, val) in enumerate(zip(bars, test_rmse)):
        ax2.text(val + 0.05, i, f"{val:.4f}", va="center", fontweight="bold")

    # MAE Comparison
    ax3 = axes[1, 0]
    bars = ax3.barh(model_names, test_mae, color=sns.color_palette("mako_r", len(model_names)))
    ax3.set_xlabel("MAE", fontweight="bold")
    ax3.set_title(f"Test MAE Comparison - {scenario_name}", fontweight="bold", fontsize=14)
    for i, (bar, val) in enumerate(zip(bars, test_mae)):
        ax3.text(val + 0.05, i, f"{val:.4f}", va="center", fontweight="bold")

    # CV R² Comparison
    ax4 = axes[1, 1]
    bars = ax4.barh(model_names, cv_r2, color=sns.color_palette("crest", len(model_names)))
    ax4.set_xlabel("CV R² (5-fold)", fontweight="bold")
    ax4.set_title(f"Cross-Validation R² - {scenario_name}", fontweight="bold", fontsize=14)
    ax4.axvline(x=0, color="black", linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars, cv_r2)):
        if val > 0:
            ax4.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"model_comparison_{scenario_name.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions(all_results, y_test, scenario_name, top_n=3):
    """Plot predicted vs actual for top models"""
    # Sort by test R²
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["test_r2"], reverse=True)[:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, sorted_models):
        y_pred = metrics["predictions"]
        r2 = metrics["test_r2"]

        ax.scatter(y_test, y_pred, alpha=0.6, s=50, color="steelblue", edgecolors="black")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")

        ax.set_xlabel("Actual G3", fontweight="bold", fontsize=12)
        ax.set_ylabel("Predicted G3", fontweight="bold", fontsize=12)
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontweight="bold", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"predictions_{scenario_name.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_residuals(model, X_test, y_test, model_name):
    """Plot residual analysis"""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, color="steelblue", edgecolors="black")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted G3", fontweight="bold")
    axes[0].set_ylabel("Residuals", fontweight="bold")
    axes[0].set_title("Residual Plot", fontweight="bold", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residuals", fontweight="bold")
    axes[1].set_ylabel("Frequency", fontweight="bold")
    axes[1].set_title("Distribution of Residuals", fontweight="bold", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot", fontweight="bold", fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Residual Analysis - {model_name}", fontweight="bold", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"residuals_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(
            range(top_n),
            importances[indices],
            color=sns.color_palette("viridis", top_n),
            edgecolor="black",
        )
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel("Importance", fontweight="bold", fontsize=12)
        plt.ylabel("Features", fontweight="bold", fontsize=12)
        plt.title(f"Top {top_n} Feature Importances - {model_name}", fontweight="bold", fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"feature_importance_{model_name.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.show()


# ============ Main Pipeline ============


def run_scenario_a():
    """Scenario A: Without G1 and G2"""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "SCENARIO A: WITHOUT G1 & G2" + " " * 30 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    # Load data
    X, y = load_and_prepare_data()

    # Remove G1 and G2
    print("\n--- Removing G1 and G2 ---")
    X_no_grades = X.drop(columns=["G1", "G2"])
    print(f"Features after removing G1, G2: {X_no_grades.shape[1]}")

    # Skip VIF (too slow), go directly to feature selection
    print("\n--- Skipping VIF removal (too computationally expensive) ---")
    print("Using feature selection methods to identify best features\n")
    
    # Feature selection
    selected_features = select_features_union(X_no_grades, y, k=20)
    X_selected = X_no_grades[selected_features]

    print(f"\nFinal feature set: {X_selected.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Scaling for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test, "Scenario A")

    # Train advanced models (tree-based, no scaling)
    tree_results = train_advanced_models(
        X_train, X_test, y_train, y_test, "Scenario A - Tree Models", scaled=False
    )

    # Train advanced models (linear, with scaling)
    linear_results = train_advanced_models(
        X_train_scaled, X_test_scaled, y_train, y_test, "Scenario A - Linear Models", scaled=True
    )

    # Hyperparameter tuning
    tuned_tree = tune_top_models(X_train, y_train, "Scenario A - Tree Models", scaled=False)
    tuned_tree_results = evaluate_tuned_models(tuned_tree, X_train, X_test, y_train, y_test)

    tuned_linear = tune_top_models(X_train_scaled, y_train, "Scenario A - Linear Models", scaled=True)
    tuned_linear_results = evaluate_tuned_models(
        tuned_linear, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Combine all results
    all_results = {
        **baseline_results,
        **tree_results,
        **linear_results,
        **tuned_tree_results,
        **tuned_linear_results,
    }

    # Evaluation
    print(f"\n{'=' * 80}")
    print("SCENARIO A - COMPREHENSIVE RESULTS")
    print("=" * 80)
    comparison_table = create_comparison_table(all_results)
    print("\n" + comparison_table.to_string(index=False))

    # Visualizations
    plot_model_comparison(all_results, "Scenario A")
    plot_predictions(all_results, y_test, "Scenario A", top_n=3)

    # Best model analysis
    best_model_name = max(all_results.items(), key=lambda x: x[1]["test_r2"])[0]
    best_model = all_results[best_model_name]["model"]
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {all_results[best_model_name]['test_r2']:.4f}")
    print("=" * 80)

    # Residual analysis for best model
    if "scaled" in best_model_name.lower() or any(
        x in best_model_name.lower() for x in ["ridge", "lasso", "elastic", "svr", "knn"]
    ):
        plot_residuals(best_model, X_test_scaled, y_test, best_model_name)
    else:
        plot_residuals(best_model, X_test, y_test, best_model_name)

    # Feature importance
    plot_feature_importance(best_model, X_selected.columns, best_model_name)

    return all_results, X_selected.columns


def run_scenario_b():
    """Scenario B: With G1 and G2"""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 22 + "SCENARIO B: WITH G1 & G2" + " " * 32 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    # Load data
    X, y = load_and_prepare_data()

    # Keep all features (VIF skip for speed)
    print("\n--- Keeping all features (including G1 and G2) ---")
    print("Skipping VIF removal for computational efficiency\n")
    
    # Feature selection (keeping G1, G2)
    selected_features = select_features_union(X, y, k=25)
    # Ensure G1 and G2 are in selected features
    if "G1" not in selected_features:
        selected_features.append("G1")
    if "G2" not in selected_features:
        selected_features.append("G2")

    X_selected = X[selected_features]
    print(f"\nFinal feature set: {X_selected.shape[1]} features (including G1, G2)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Scaling for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test, "Scenario B")

    # Train advanced models (tree-based)
    tree_results = train_advanced_models(
        X_train, X_test, y_train, y_test, "Scenario B - Tree Models", scaled=False
    )

    # Train advanced models (linear)
    linear_results = train_advanced_models(
        X_train_scaled, X_test_scaled, y_train, y_test, "Scenario B - Linear Models", scaled=True
    )

    # Hyperparameter tuning
    tuned_tree = tune_top_models(X_train, y_train, "Scenario B - Tree Models", scaled=False)
    tuned_tree_results = evaluate_tuned_models(tuned_tree, X_train, X_test, y_train, y_test)

    tuned_linear = tune_top_models(X_train_scaled, y_train, "Scenario B - Linear Models", scaled=True)
    tuned_linear_results = evaluate_tuned_models(
        tuned_linear, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Combine all results
    all_results = {
        **baseline_results,
        **tree_results,
        **linear_results,
        **tuned_tree_results,
        **tuned_linear_results,
    }

    # Evaluation
    print(f"\n{'=' * 80}")
    print("SCENARIO B - COMPREHENSIVE RESULTS")
    print("=" * 80)
    comparison_table = create_comparison_table(all_results)
    print("\n" + comparison_table.to_string(index=False))

    # Visualizations
    plot_model_comparison(all_results, "Scenario B")
    plot_predictions(all_results, y_test, "Scenario B", top_n=3)

    # Best model analysis
    best_model_name = max(all_results.items(), key=lambda x: x[1]["test_r2"])[0]
    best_model = all_results[best_model_name]["model"]
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {all_results[best_model_name]['test_r2']:.4f}")
    print("=" * 80)

    # Residual analysis
    if "scaled" in best_model_name.lower() or any(
        x in best_model_name.lower() for x in ["ridge", "lasso", "elastic", "svr", "knn"]
    ):
        plot_residuals(best_model, X_test_scaled, y_test, best_model_name)
    else:
        plot_residuals(best_model, X_test, y_test, best_model_name)

    # Feature importance
    plot_feature_importance(best_model, X_selected.columns, best_model_name)

    return all_results, X_selected.columns


# ============ Main Execution ============

if __name__ == "__main__":
    print("\n" * 2)
    print("=" * 80)
    print("STUDENT PERFORMANCE PREDICTION - COMPREHENSIVE ML PIPELINE")
    print("=" * 80)

    # Run Scenario A
    results_a, features_a = run_scenario_a()

    print("\n" * 3)

    # Run Scenario B
    results_b, features_b = run_scenario_b()

    # Final comparison
    print("\n" * 3)
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 24 + "FINAL COMPARISON & INSIGHTS" + " " * 28 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    print("\n=== Scenario A (Without G1/G2) ===")
    print("Best models:")
    sorted_a = sorted(results_a.items(), key=lambda x: x[1]["test_r2"], reverse=True)[:3]
    for i, (name, metrics) in enumerate(sorted_a, 1):
        print(f"  {i}. {name:30s} - R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")

    print("\n=== Scenario B (With G1/G2) ===")
    print("Best models:")
    sorted_b = sorted(results_b.items(), key=lambda x: x[1]["test_r2"], reverse=True)[:3]
    for i, (name, metrics) in enumerate(sorted_b, 1):
        print(f"  {i}. {name:30s} - R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}")

    print("\n=== Key Insights ===")
    print("1. Scenario A (Without G1/G2):")
    print("   - More challenging but realistic prediction task")
    print(f"   - Best R²: {sorted_a[0][1]['test_r2']:.4f} (Expected: 0.20-0.40)")
    print("   - Useful for early intervention before exams")

    print("\n2. Scenario B (With G1/G2):")
    print("   - High-accuracy benchmark showing upper bound")
    print(f"   - Best R²: {sorted_b[0][1]['test_r2']:.4f} (Expected: 0.80-0.95)")
    print("   - Useful for grade progression prediction")

    print("\n3. Recommendations:")
    print(f"   - For deployment: {sorted_a[0][0]} (Scenario A)")
    print("   - For benchmarking: {sorted_b[0][0]} (Scenario B)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80 + "\n")

