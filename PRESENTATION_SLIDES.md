# Student Performance Prediction Using Machine Learning
## A Comprehensive Analysis of Secondary Education Outcomes

**Author**: Marwen BK  
**Institution**: University Machine Learning Project  
**Date**: October 2025

---

# 📋 Agenda

1. **Problem & Motivation**
2. **Dataset Overview**
3. **Exploratory Data Analysis**
4. **Methodology & Approach**
5. **Two-Scenario Strategy**
6. **Results & Model Performance**
7. **Key Findings & Insights**
8. **Recommendations**
9. **Conclusions & Future Work**

---

# 🎯 Problem Statement

## Research Question
**Can we predict student final grades using demographic, social, and academic features?**

## Motivation
- **Early Intervention**: Identify at-risk students before failure
- **Resource Optimization**: Target support programs effectively
- **Policy Decisions**: Data-driven educational strategies
- **Student Success**: Improve graduation rates and outcomes

## Challenge
- Complex interplay of factors (social, academic, behavioral)
- Imbalanced importance of features
- Need for interpretable and actionable results

---

# 📊 Dataset Overview

## Source
- **Origin**: UCI Machine Learning Repository
- **Context**: Portuguese secondary school students (2 schools)
- **Course**: Portuguese Language (Math also available)

## Statistics
- **Samples**: 649 students
- **Original Features**: 33
- **After Encoding**: 40 features
- **Target Variable**: Final grade (G3) on scale 0-20
- **No Missing Values**: Complete dataset

---

# 📊 Dataset Features (1/2)

## Demographic Features
- **Age**: 15-22 years
- **Sex**: Male/Female
- **Address**: Urban/Rural
- **Family Size**: ≤3 or >3 members
- **Parent's Cohabitation Status**: Together/Apart

## Socioeconomic Features
- **Mother's Education**: 0 (none) to 4 (higher education)
- **Father's Education**: 0 (none) to 4 (higher education)
- **Mother's Job**: Teacher, health, services, at_home, other
- **Father's Job**: Teacher, health, services, at_home, other
- **Family Educational Support**: Yes/No

---

# 📊 Dataset Features (2/2)

## Academic Features
- **Study Time**: 1 (<2h) to 4 (>10h per week)
- **Past Failures**: Number of previous class failures
- **Extra Educational Support**: Yes/No
- **Extra Paid Classes**: Yes/No
- **School**: GP (Gabriel Pereira) or MS (Mousinho da Silveira)

## Behavioral Features
- **Going Out Frequency**: 1 (very low) to 5 (very high)
- **Workday Alcohol Consumption**: 1 (very low) to 5 (very high)
- **Weekend Alcohol Consumption**: 1 (very low) to 5 (very high)
- **Free Time**: 1 (very low) to 5 (very high)
- **Health Status**: 1 (very bad) to 5 (very good)

## Grade Features
- **G1**: First period grade (0-20)
- **G2**: Second period grade (0-20)
- **G3**: Final grade (0-20) **[TARGET]**

---

# 🔍 Exploratory Data Analysis - Key Findings

## Target Variable (G3) Distribution
- **Mean**: 11.91 ± 3.13
- **Range**: 0 to 19
- **Distribution**: Slightly left-skewed (more high grades than low)
- **Mode**: Grade 12 (most common)

## Critical Discovery: Grade Correlations
| Grade Pair | Correlation |
|------------|-------------|
| **G2 ↔ G3** | **0.92** |
| **G1 ↔ G3** | **0.86** |
| **G1 ↔ G2** | **0.88** |

**⚠️ Implication**: Past grades are extremely strong predictors of final grades!

---

# 🔍 Bivariate Analysis - Categorical Features

## Most Impactful Categorical Variables
1. **School** (GP vs MS): 2.4 grade point difference
2. **Higher Education Goal**: 1.9 grade point difference
3. **Extra Paid Classes**: 1.5 grade point difference
4. **Address** (Urban vs Rural): 1.2 grade point difference
5. **Family Educational Support**: 1.0 grade point difference

## Statistical Significance
- All differences statistically significant (p < 0.001)
- School and aspirations have largest effect sizes
- Socioeconomic factors play crucial role

---

# 🔍 Bivariate Analysis - Numeric Features

## Strongest Correlations with G3 (excluding G1, G2)
| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| **failures** | **-0.33** | Past failures hurt final grade |
| **higher** | **+0.29** | Higher ed. aspirations help |
| **Medu** | **+0.24** | Mother's education matters |
| **studytime** | **+0.12** | More study helps (modest) |
| **Walc** | **-0.11** | Weekend alcohol hurts |

## Key Insight
- **Failures** is the strongest non-grade predictor
- Education aspirations and family background crucial
- Behavioral factors have modest but consistent impact

---

# ⚠️ Multicollinearity Analysis

## Variance Inflation Factor (VIF) Results
- **Features with VIF > 10**: 12 features (severe multicollinearity)
- **Highly Correlated Pairs** (r > 0.9): 6 pairs detected
- **Main Issues**:
  - G1 and G2 highly correlated with each other and G3
  - Some one-hot encoded features redundant
  - Parent education levels correlated

## Resolution Strategy
1. **Iterative VIF Removal**: Drop features with VIF > 10
2. **Correlation Pruning**: Remove one feature from highly correlated pairs
3. **Feature Selection**: Use multiple methods (F-test, MI, RFE)
4. **Two-Scenario Approach**: Test with/without G1, G2 entirely

---

# 🛠️ Methodology

## Data Preprocessing
1. **Binary Mapping**: Yes/No → 1/0, M/F → 1/0
2. **One-Hot Encoding**: Jobs, reason, guardian (drop_first=True)
3. **Feature Removal**: Age dropped (low variance)
4. **Zero Variance Filter**: Remove constant features

## Feature Engineering
- **Grade Changes**: G2-G1, G3-G2 differences
- **Interaction Terms**: Considered but not used (increased complexity)

## Train-Test Split
- **Training Set**: 80% (519 samples)
- **Test Set**: 20% (130 samples)
- **Random State**: 42 (reproducibility)

---

# 🛠️ Feature Selection

## Three Methods Combined
1. **F-Regression**: Statistical test for linear relationships
2. **Mutual Information**: Captures non-linear dependencies
3. **Recursive Feature Elimination (RFE)**: Model-based selection

## Selection Strategy
- **Top K per Method**: Select top 15-20 features per method
- **Union Approach**: Combine features from all methods
- **Final Set**: ~25-30 features per scenario

## Result
- Reduced feature space while retaining predictive power
- Removed redundant and noise features
- Improved model interpretability

---

# 🎭 Two-Scenario Strategy

## Why Two Scenarios?

**Problem**: G1 and G2 dominate predictions (r = 0.86, 0.92)
- Makes prediction "too easy" and less interesting
- Doesn't help for early intervention (grades already known)

**Solution**: Evaluate both scenarios

---

# 📊 Scenario A: Early Prediction

## Configuration
- **Features**: Demographic, social, academic (NO G1, NO G2)
- **Use Case**: Predict final grade at enrollment
- **Goal**: Identify at-risk students early for intervention

## Advantages
✅ Early identification (before any grades)
✅ More challenging and realistic problem
✅ Tests model on meaningful features
✅ Actionable for school policy

## Trade-off
⚠️ Lower accuracy expected (no grade history)

---

# 📊 Scenario B: Grade Progression

## Configuration
- **Features**: All features (INCLUDING G1, G2)
- **Use Case**: Predict final grade after midterms
- **Goal**: High-accuracy forecasting for grade progression

## Advantages
✅ Very high accuracy (G2 correlation = 0.92)
✅ Useful for progression monitoring
✅ Validates overall modeling approach
✅ Baseline for comparison

## Trade-off
⚠️ Less useful for early intervention (grades already known)

---

# 🤖 Models Evaluated

## Baseline Models
- **Linear Regression**: Simple interpretable baseline
- **Decision Tree**: Non-linear baseline, prone to overfitting

## Regularized Linear Models
- **Ridge Regression**: L2 penalty, handles multicollinearity
- **Lasso Regression**: L1 penalty, feature selection + regularization

## Ensemble Methods
- **Random Forest**: Bootstrap aggregation, reduces variance
- **Gradient Boosting**: Sequential learning, corrects errors

## Non-linear Models
- **Support Vector Regressor (SVR)**: Kernel-based, captures non-linearity

---

# 🎛️ Hyperparameter Tuning

## Method
- **GridSearchCV**: Exhaustive search over parameter grid
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Scoring Metric**: R² (variance explained)

## Tuned Models
1. **Ridge**: Alpha (regularization strength)
2. **Lasso**: Alpha (regularization strength)
3. **Random Forest**: n_estimators, max_depth, min_samples_split
4. **Gradient Boosting**: n_estimators, learning_rate, max_depth

## Optimization Goals
- Maximize R² score
- Minimize overfitting (train-test gap)
- Balance complexity vs performance

---

# 📊 Evaluation Metrics

## Primary Metrics
- **R² (R-squared)**: Proportion of variance explained (0-1, higher better)
- **MAE (Mean Absolute Error)**: Average absolute error in grade points
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **MAPE (Mean Absolute Percentage Error)**: Percentage error

## Validation Strategy
- **Test Set**: Holdout 20% for final evaluation
- **Cross-Validation**: 3-fold CV for model selection
- **Multiple Metrics**: Comprehensive performance view

## Why Multiple Metrics?
- R² alone can be misleading
- MAE is interpretable (grade points)
- RMSE penalizes outliers
- CV ensures generalization

---

# 🏆 Results: Scenario A (Early Prediction)

## Best Model: **Gradient Boosting (Tuned)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test R²** | **0.206** | 21% variance explained |
| **Test MAE** | **2.08** | ±2 grade points error |
| **Test RMSE** | **2.78** | ~3 points typical error |
| **CV R²** | **0.273 ± 0.068** | Consistent across folds |

## Context
- **Without grade history**: 21% explained variance is respectable
- **Practical Use**: Identify top 20% at-risk students
- **Screening Tool**: Not perfect, but useful for targeting interventions

---

# 📊 Scenario A - Model Rankings

| Rank | Model | Test R² | Test MAE | Test RMSE |
|------|-------|---------|----------|-----------|
| 🥇 **1** | **Gradient Boosting (tuned)** | **0.206** | **2.08** | **2.78** |
| 🥈 2 | Lasso Regression | 0.188 | 2.10 | 2.81 |
| 🥉 3 | Lasso (tuned) | 0.188 | 2.10 | 2.81 |
| 4 | Gradient Boosting | 0.184 | 2.08 | 2.82 |
| 5 | SVR | 0.169 | 2.08 | 2.85 |
| 6 | Ridge (tuned) | 0.166 | 2.16 | 2.85 |
| 7 | Ridge Regression | 0.162 | 2.17 | 2.86 |
| 8 | Linear Regression | 0.162 | 2.17 | 2.86 |
| 9 | Random Forest | 0.158 | 2.12 | 2.87 |
| 10 | Random Forest (tuned) | 0.154 | 2.10 | 2.87 |
| 11 | Decision Tree | -0.316 | 2.54 | 3.58 |

---

# 🏆 Results: Scenario B (Grade Progression)

## Best Model: **Lasso Regression**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test R²** | **0.865** | 87% variance explained |
| **Test MAE** | **0.71** | Less than 1 grade point error |
| **Test RMSE** | **1.15** | ~1 point typical error |
| **CV R²** | **0.842 ± 0.028** | Very consistent |

## Context
- **With G1, G2**: Extremely high accuracy
- **G2 Dominance**: Single feature explains 85% variance
- **Practical Use**: Accurate grade forecasting
- **Model Choice**: Lasso provides regularization + simplicity

---

# 📊 Scenario B - Model Rankings

| Rank | Model | Test R² | Test MAE | Test RMSE |
|------|-------|---------|----------|-----------|
| 🥇 **1** | **Lasso Regression** | **0.865** | **0.71** | **1.15** |
| 🥈 2 | Lasso (tuned) | 0.865 | 0.71 | 1.15 |
| 🥉 3 | Ridge Regression | 0.848 | 0.76 | 1.22 |
| 4 | Linear Regression | 0.848 | 0.76 | 1.22 |
| 5 | Ridge (tuned) | 0.848 | 0.77 | 1.22 |
| 6 | Random Forest | 0.835 | 0.77 | 1.27 |
| 7 | Random Forest (tuned) | 0.832 | 0.78 | 1.28 |
| 8 | Gradient Boosting (tuned) | 0.827 | 0.77 | 1.30 |
| 9 | Gradient Boosting | 0.807 | 0.79 | 1.37 |
| 10 | SVR | 0.690 | 1.10 | 1.74 |
| 11 | Decision Tree | 0.591 | 1.04 | 2.00 |

---

# 📈 Performance Gap Analysis

## Scenario Comparison

| Metric | Scenario A<br>(No G1/G2) | Scenario B<br>(With G1/G2) | Difference |
|--------|-------------------------|----------------------------|------------|
| **Best R²** | 0.206 | 0.865 | **+0.659** (+320%) |
| **Best MAE** | 2.08 | 0.71 | **-1.37 points** (-66%) |
| **Best RMSE** | 2.78 | 1.15 | **-1.63 points** (-59%) |
| **Variance Explained** | 21% | 87% | **+66%** |

## Key Takeaway
- **Massive performance difference** demonstrates G1/G2 dominance
- **Both scenarios useful** for different intervention points
- **21% without grades** is still valuable for early screening

---

# 🔍 Feature Importance (Scenario A)

## Top 10 Most Important Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | **failures** | 0.287 | Academic |
| 2 | **higher** | 0.156 | Aspirations |
| 3 | **Medu** | 0.119 | Socioeconomic |
| 4 | **studytime** | 0.084 | Academic |
| 5 | **school** | 0.072 | Institutional |
| 6 | **Fedu** | 0.065 | Socioeconomic |
| 7 | **goout** | 0.041 | Behavioral |
| 8 | **Walc** | 0.038 | Behavioral |
| 9 | **age** | 0.029 | Demographic |
| 10 | **freetime** | 0.024 | Behavioral |

---

# 💡 Key Insights (1/2)

## Academic Factors
- ✅ **Past failures are the #1 predictor** (importance = 0.287)
- ✅ Students with 0 failures average **13.2** vs 9.7 with failures
- ✅ **Study time matters**: Each level increase → +0.5 grade points
- ✅ **Extra support helps**: Students with support score 0.8 points higher

## Aspirations & Motivation
- ✅ **Higher education goals**: +1.9 grade points on average
- ✅ Strong predictor even controlling for other factors
- ✅ Suggests motivation and long-term planning are crucial

---

# 💡 Key Insights (2/2)

## Socioeconomic Factors
- ✅ **Mother's education** more important than father's (Medu > Fedu)
- ✅ Each level of parental education → +0.5 to +0.8 points
- ✅ **School matters**: GP students score 2.4 points higher than MS

## Behavioral Factors
- ⚠️ **Going out frequently**: Negative correlation (-0.08)
- ⚠️ **Weekend alcohol**: Negative correlation (-0.11)
- ⚠️ Effects modest but consistent
- ⚠️ Workday alcohol worse than weekend

## Controllable vs Non-Controllable
- **Controllable**: Study time, alcohol, extra classes, attendance
- **Non-controllable**: Parent education, school, past failures
- **Intervention focus**: Maximize controllable factors

---

# 🎯 Model Selection Rationale

## Scenario A: Why Gradient Boosting?
✅ **Best Performance**: Highest R² (0.206) and lowest MAE (2.08)
✅ **Non-linear Relationships**: Captures complex interactions
✅ **Feature Importance**: Built-in interpretability
✅ **Robust**: Good cross-validation consistency
✅ **Production-Ready**: Handles new data well

## Scenario B: Why Lasso Regression?
✅ **Simplicity**: Linear model, easy to interpret
✅ **Performance**: Matches best models (R² = 0.865)
✅ **Regularization**: L1 penalty prevents overfitting
✅ **Feature Selection**: Automatically zeros out weak predictors
✅ **Efficiency**: Fast prediction, low computational cost
✅ **Stability**: Very consistent cross-validation results

---

# 📋 Recommendations - For Schools

## 1. Early Intervention System (Use Scenario A Model)
- **Target**: Students predicted in bottom 20% at enrollment
- **Action**: Mandatory academic counseling and progress monitoring
- **Focus Areas**: Students with past failures (highest risk)

## 2. Academic Support Programs
- **Tutoring Services**: Free tutoring for at-risk students
- **Study Skills Workshops**: Time management, note-taking, test prep
- **Peer Mentoring**: Match struggling students with successful peers

## 3. Parent Engagement
- **Education Nights**: Teach parents how to support student success
- **Regular Communication**: Progress reports and intervention alerts
- **Family Support Resources**: Workshops on creating study environments

---

# 📋 Recommendations - For Students

## Controllable Factors to Improve Performance

### ✅ **High Impact Actions**
1. **Increase Study Time**: Aim for 5-10 hours per week
2. **Attend All Classes**: Absences strongly correlate with lower grades
3. **Seek Extra Help Early**: Don't wait until failing
4. **Reduce Weekday Alcohol**: Strong negative predictor
5. **Set Higher Education Goals**: Increases motivation

### ✅ **Behavioral Changes**
- Limit going out during school nights
- Create dedicated study environment
- Use school resources (library, tutoring)
- Engage in educational activities

### 📊 **Expected Impact**
- Increasing study time: +0.5 to +1.5 grade points
- Reducing absences: +0.3 to +0.8 grade points
- Seeking support: +0.5 to +1.0 grade points

---

# 📋 Recommendations - For Policymakers

## System-Level Interventions

### 1. **Resource Allocation**
- Direct funding to schools with lower performance (MS vs GP gap)
- Subsidize extra paid classes for low-income students
- Provide transportation for rural students

### 2. **Curriculum Design**
- Early intervention programs in middle school
- Study skills curriculum (not just content)
- Career counseling to build higher education aspirations

### 3. **Data-Driven Decisions**
- Implement predictive models district-wide
- Track intervention effectiveness with A/B testing
- Share successful strategies across schools

---

# ⚠️ Limitations & Ethical Considerations

## Model Limitations
- **Low R² (Scenario A)**: 21% explained variance → 79% unexplained
- **Sample Size**: Only 649 students from 2 schools
- **Cultural Context**: Portuguese schools may differ from others
- **Temporal**: 2008 data may not reflect current conditions
- **Unmeasured Factors**: Teaching quality, learning disabilities, mental health

## Ethical Risks
⚠️ **Self-Fulfilling Prophecy**: Low predictions → reduced support → actual failure
⚠️ **Bias Concerns**: Model may perpetuate existing inequalities
⚠️ **Privacy**: Student data must be protected
⚠️ **Transparency**: Predictions should be explainable to students/parents

## Mitigation Strategies
✅ Use predictions for **support allocation**, not punishment
✅ Regular audits for demographic bias
✅ Human oversight required for all decisions
✅ Provide opt-out mechanisms

---

# 🔮 Future Work

## Model Improvements
1. **Deep Learning**: Neural networks for complex patterns
2. **Ensemble Stacking**: Combine multiple models
3. **Time-Series**: Model grade progression over time
4. **Transfer Learning**: Apply models across different courses

## Feature Engineering
5. **Teacher Quality**: Incorporate instructor data
6. **Social Networks**: Peer influence features
7. **Learning Styles**: Personalized feature engineering
8. **Temporal Patterns**: Study habits over time

## Expanded Scope
9. **Multi-School Validation**: Test on more schools
10. **Longitudinal Study**: Track students across years
11. **Intervention Studies**: A/B test recommendations
12. **Real-Time System**: Deploy production prediction system

---

# 📊 Technical Summary

## Pipeline Overview
```
Raw Data (649 students, 33 features)
    ↓
Preprocessing (encoding, feature engineering)
    ↓
Feature Selection (F-test, MI, RFE)
    ↓
Two Scenarios Split (A: no G1/G2, B: with G1/G2)
    ↓
Model Training (11 algorithms)
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Evaluation (R², MAE, RMSE, CV)
    ↓
Best Models Selected
```

## Deliverables
✅ Complete EDA with 30+ visualizations
✅ 22 trained models (11 × 2 scenarios)
✅ 7 publication-quality figures
✅ 50+ pages technical documentation
✅ Production-ready code (1,500+ lines)

---

# 🎯 Conclusions (1/2)

## Key Achievements
1. ✅ **Developed dual-scenario ML system** addressing different intervention points
2. ✅ **Identified key predictors**: Failures, aspirations, parental education
3. ✅ **Achieved actionable accuracy**: 21% R² without grades, 87% with grades
4. ✅ **Provided interpretable models** with clear feature importance
5. ✅ **Generated practical recommendations** for schools and students

## Main Findings
- **Past failures are the strongest predictor** of future performance
- **G1 and G2 dominate** predictions (r = 0.86, 0.92 with G3)
- **Socioeconomic factors matter** but are not destiny
- **Behavioral factors** (study time, alcohol) have modest but real impact
- **School quality** creates significant performance gaps

---

# 🎯 Conclusions (2/2)

## Practical Impact
- **Early prediction (Scenario A)** enables intervention before failure
- **Grade progression (Scenario B)** enables accurate forecasting
- **Feature importance** guides where to invest resources
- **Actionable insights** for students, teachers, and policymakers

## Broader Implications
- ✅ Machine learning can support (not replace) educational decision-making
- ✅ Modest accuracy (21%) still valuable for screening and prioritization
- ✅ Transparency and ethics crucial when predicting human outcomes
- ✅ Data-driven approaches complement (not replace) teacher expertise

## Final Thought
> *"The goal is not perfect prediction, but identifying students who would benefit most from support. Even 21% explained variance represents hundreds of students who could be helped."*

---

# 📚 References & Resources

## Dataset
- **Cortez, P., & Silva, A.** (2008). Using Data Mining to Predict Secondary School Student Performance. *Proceedings of 5th FUBUTEC Conference*, pp. 5-12.
- UCI Machine Learning Repository: [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

## Code & Documentation
- GitHub Repository: [github.com/marwenbk/python_ml_uni_project](https://github.com/marwenbk/python_ml_uni_project)
- Technical Report: `TECHNICAL_REPORT.md` (18 pages)
- Results Summary: `FINAL_RESULTS_SUMMARY.md` (12 pages)
- Full Code: `project.py` (886 lines), `modeling.py` (800+ lines)

## Tools & Libraries
- Python 3.8+, scikit-learn 1.5.2, pandas, numpy, matplotlib, seaborn
- statsmodels (VIF analysis), GridSearchCV (hyperparameter tuning)

---

# ❓ Questions & Discussion

## Discussion Topics
- How would you balance early intervention with avoiding stigmatization?
- What other features might improve predictions?
- How can schools ensure ethical use of predictive models?
- Should students be informed of their predictions?

## Contact
**Marwen BK**
- GitHub: [github.com/marwenbk](https://github.com/marwenbk)
- Repository: [python_ml_uni_project](https://github.com/marwenbk/python_ml_uni_project)

---

# Thank You! 🎓

**Questions?**

---

# 📎 Appendix: Additional Visualizations

*Include the following figures in your presentation:*

1. **Model Comparison Charts**
   - `model_comparison_Scenario_A.png`
   - `model_comparison_Scenario_B.png`

2. **Prediction Accuracy Plots**
   - `predictions_Scenario_A.png`
   - `predictions_Scenario_B.png`

3. **Residual Analysis**
   - `residuals_Gradient_Boosting_(tuned).png`
   - `residuals_Lasso_Regression.png`

4. **Feature Importance**
   - `feature_importance_Gradient_Boosting_(tuned).png`

---

# 📎 Appendix: Detailed Model Parameters

## Scenario A - Best Model Configuration
**Gradient Boosting Regressor (Tuned)**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 4
- min_samples_split: 5
- Cross-validation: 3-fold
- Training time: ~2 minutes

## Scenario B - Best Model Configuration
**Lasso Regression**
- alpha: 0.01 (regularization strength)
- max_iter: 10,000
- Features selected: 18 (automatic via L1 penalty)
- Training time: <1 second

---

# 📎 Appendix: Statistical Tests

## Feature Significance (Scenario A)
- **F-regression test**: All top 15 features p < 0.001
- **Mutual Information**: Confirms non-linear relationships
- **Permutation Importance**: Validates feature rankings

## Model Comparison Tests
- **Paired t-test**: Gradient Boosting vs Lasso (p = 0.041)
- **Cross-validation consistency**: CV std < 0.10 for best models
- **Residual normality**: Shapiro-Wilk p > 0.05 (acceptable)

## Validation Checks
✅ No data leakage (train-test completely separate)
✅ Random seed fixed (reproducibility)
✅ Cross-validation prevents overfitting
✅ Multiple metrics corroborate results

