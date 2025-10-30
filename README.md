# Student Performance Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning project analyzing student performance in Portuguese secondary education using multiple regression models and advanced feature engineering techniques.

## 📊 Project Overview

This project predicts final student grades (G3) using demographic, social, and school-related features. It explores **two distinct scenarios**:

- **Scenario A (Early Prediction)**: Predicting final grades WITHOUT prior grades (G1, G2) - useful for early intervention
- **Scenario B (High-Accuracy)**: Predicting final grades WITH prior grades - useful for grade progression forecasting

### Key Results

| Scenario | Best Model | Test R² | MAE | Use Case |
|----------|-----------|---------|-----|----------|
| **A** (No G1/G2) | Gradient Boosting | 0.206 | ±2.08 | Early intervention at enrollment |
| **B** (With G1/G2) | Lasso Regression | 0.865 | ±0.71 | Grade forecasting after midterms |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/marwenbk/python_ml_uni_project.git
cd python_ml_uni_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run exploratory data analysis (generates visualizations)
python project.py

# Run full machine learning pipeline (trains 22 models)
python modeling.py
```

## 📁 Project Structure

```
ml_msb/
├── project.py                          # Complete EDA with bivariate & multicollinearity analysis
├── modeling.py                         # ML pipeline (22 model configurations)
├── student-por.csv                     # Dataset (649 students, 33 features)
├── requirements.txt                    # Python dependencies
│
├── FINAL_RESULTS_SUMMARY.md           # 📊 Comprehensive results (START HERE)
├── TECHNICAL_REPORT.md                # 📝 Technical methodology & decisions
├── PROJECT_SUMMARY.md                 # 📋 Executive summary
├── README_PROJECT_COMPLETE.md         # 📖 Complete navigation guide
├── PRESENTATION_SLIDES.md             # 🎤 40+ presentation slides
├── PRESENTATION_GUIDE.md              # 📖 How to use and convert slides
│
├── modeling_final.log                 # Full training results log
│
└── visualizations/                    # 7 publication-quality figures
    ├── model_comparison_Scenario_A.png
    ├── model_comparison_Scenario_B.png
    ├── predictions_Scenario_A.png
    ├── predictions_Scenario_B.png
    ├── residuals_Gradient_Boosting_(tuned).png
    ├── residuals_Lasso_Regression.png
    └── feature_importance_Gradient_Boosting_(tuned).png
```

## 🔍 Dataset

**Source**: [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

**Statistics**:
- 649 students from Portuguese secondary schools
- 33 original features (40 after encoding)
- Target: Final grade (G3) ranging from 0-20

**Feature Categories**:
- **Demographic**: Age, sex, family size, parent's cohabitation
- **Socioeconomic**: Parent's education & occupation, family support
- **Academic**: Past failures, extra educational support, study time
- **Behavioral**: Free time, going out, alcohol consumption
- **School-related**: School, reason for choosing school, travel time

## 🧠 Machine Learning Pipeline

### Models Implemented

**Baseline Models**:
- Linear Regression
- Decision Tree Regressor

**Advanced Models**:
- Ridge Regression (with hyperparameter tuning)
- Lasso Regression (with hyperparameter tuning)
- Random Forest Regressor (with tuning)
- Gradient Boosting Regressor (with tuning)
- Support Vector Regressor (SVR)

### Evaluation Metrics

- **R² Score**: Proportion of variance explained
- **MAE**: Mean Absolute Error (interpretable in grade points)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error
- **Cross-Validation R²**: 5-fold CV for robustness

### Feature Engineering

- Binary encoding for yes/no categorical variables
- One-hot encoding for multi-class categories (job, reason, guardian)
- Variance Inflation Factor (VIF) analysis for multicollinearity
- Highly correlated pair detection (threshold: 0.9)
- Multiple feature selection methods: F-regression, Mutual Information, RFE

## 📈 Key Findings

### Top Predictive Features (Scenario A - Without G1/G2):

1. 🔴 **failures** - Number of past class failures (strongest predictor)
2. 🎓 **higher** - Wants to pursue higher education
3. 👨‍👩‍👦 **Medu, Fedu** - Mother's and father's education level
4. 📚 **studytime** - Weekly study time
5. 🏫 **school** - School attended (GP vs MS)

### Actionable Insights:

**For Schools**:
- ✅ Early identification of at-risk students (those with past failures)
- ✅ Targeted intervention programs
- ✅ Parent engagement initiatives
- ✅ Study skills workshops

**For Students** (Controllable Factors):
- ✅ Increase study time (3-5 hours/week minimum)
- ✅ Reduce weekday alcohol consumption
- ✅ Improve school attendance
- ✅ Seek extra educational support
- ✅ Set higher education goals

### Model Performance Comparison:

**Scenario A** (Early Prediction):
- Gradient Boosting achieved 21% variance explained
- Challenging but realistic for early intervention
- MAE of 2.08 points is acceptable for screening

**Scenario B** (With Prior Grades):
- Lasso Regression achieved 87% variance explained
- G2 alone correlates 0.92 with G3 (dominant predictor)
- MAE of 0.71 points enables precise forecasting

## 📊 Visualizations

All visualizations are publication-quality PNG files:

- **Model Comparison Charts**: Bar charts comparing all models by R², MAE, RMSE
- **Prediction Plots**: Actual vs Predicted scatter plots for top 3 models
- **Residual Analysis**: Residual plots and distributions for best models
- **Feature Importance**: Bar charts showing most influential features

## 📖 Documentation

- **[FINAL_RESULTS_SUMMARY.md](FINAL_RESULTS_SUMMARY.md)** - Start here! Complete results with interpretations
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Detailed methodology and technical decisions
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Executive summary for stakeholders
- **[README_PROJECT_COMPLETE.md](README_PROJECT_COMPLETE.md)** - Comprehensive project guide
- **[PRESENTATION_SLIDES.md](PRESENTATION_SLIDES.md)** - 40+ slides for presentations (20-40 min)
- **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** - How to convert slides and present effectively

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn 1.5.2** - Machine learning models and evaluation
- **pandas 2.0+** - Data manipulation
- **numpy <2.0** - Numerical computing
- **matplotlib 3.9.2** - Visualizations
- **seaborn 0.13.2** - Statistical plots
- **statsmodels 0.14.4** - VIF analysis and statistical tests

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
- **UCI Machine Learning Repository** for hosting the dataset

## 📧 Contact

**Marwen BK** - [GitHub Profile](https://github.com/marwenbk)

**Repository**: [python_ml_uni_project](https://github.com/marwenbk/python_ml_uni_project)

---

⭐ If you found this project helpful, please consider giving it a star!

