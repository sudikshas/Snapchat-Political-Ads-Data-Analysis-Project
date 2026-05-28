
# Snapchat Political Ad Engagement — Predictive Modeling & Algorithmic Fairness Analysis

*Python · Pandas · Scikit-learn · Logistic Regression · Permutation Testing · Matplotlib · Seaborn*

---

## Problem

Political advertising on social media operates as a black box — campaigns spend millions without
understanding what actually drives reach, and regulators struggle to audit whether platforms treat
different types of content equitably. This project reverse-engineers engagement patterns in
Snapchat's political ad dataset to answer two questions:

1. **What features predict whether a political ad achieves high vs. low viewership?**
2. **Does a predictive model treat high- and low-reach ad groups fairly (equal false positive rates)?**

---

## Dataset

**Source:** Snapchat Political Ads Library (2018–2019), publicly released under political ad
transparency regulations.

| Attribute | Detail |
|---|---|
| Records | ~2,000+ political ad impressions |
| Time span | 2018–2019 U.S. election cycle |
| Key features | Ad spend, targeting geography, billing city/state, start time, view counts |
| Target variable | Binary — high vs. low view count (median split) |
| Notable challenge | Missing data across multiple columns requiring formal missingness analysis |

---

## Approach

### 1. Missingness Analysis
Formally classified missing values across columns as MCAR, MAR, or NMAR using permutation tests
with Total Variation Distance (TVD). Found that `LocationType` missingness is **MAR-dependent on
`City`** (p = 0.013), informing imputation strategy and model feature selection.

### 2. Feature Engineering
- Parsed raw billing address strings to extract city and state fields
- Mapped city names to IANA timezones using `pytz` to convert UTC timestamps into local time-of-day
- Derived a `part_of_day` categorical feature (morning / afternoon / evening / night) to capture
  release timing effects across 115+ cities globally

### 3. Hypothesis Testing
Ran two permutation tests (n = 10,000 permutations each, α = 0.05) to confirm that observed
differences in ad spend and release time between high- and low-reach ads were statistically
significant and not due to random chance (both tests: p ≈ 0.0).

### 4. Predictive Modeling
Built a binary classification pipeline using `sklearn.pipeline.Pipeline` and `ColumnTransformer`:

- **Preprocessing:** `OneHotEncoder` for categorical features, `StandardScaler` for numerics
- **Model:** Logistic Regression
- **Tuning:** `GridSearchCV` with 5-fold cross-validation over regularization strength (`C`)
- **Baseline accuracy:** 53% (majority-class dummy classifier)
- **Final model accuracy:** ~91%

### 5. Fairness Evaluation
Evaluated the final model for **false positive rate (FPR) parity** across view-count groups using
a permutation test on observed FPR differences. Result: p ≈ 0.82 — no statistically significant
disparity in misclassification rates between groups, confirming the model does not disproportionately
penalize either class.

---

## Results

| Metric | Value |
|---|---|
| Baseline accuracy | 53% |
| Final model accuracy | ~91% |
| FPR parity p-value | 0.82 (no significant disparity) |
| Top predictive features | Ad spend, release time-of-day, targeting geography |
| Missingness test (LocationType ~ City) | p = 0.013 (MAR confirmed) |

---

## Key Insights & Learnings

- **Ad spend and timing are strong signals.** Even controlling for other features, when and how
  much a campaign spends are the dominant predictors of high reach — not content or targeting
  geography alone.

- **Missingness is not random.** Treating all missing values as MCAR would have introduced
  systematic bias. Formal missingness testing is a prerequisite to credible modeling, not an
  afterthought.

- **Fairness evaluation requires explicit design.** The model's high accuracy masked the question
  of equity entirely. Only by separately measuring FPR across groups could we confirm the
  classifier behaves consistently — a step often skipped in practice.

- **Time zone normalization matters at scale.** UTC timestamps without localization produce
  meaningless "time of day" features for geographically distributed datasets. Building a
  city-to-timezone mapping pipeline was essential to extracting real signal.

---

## Business Impact

| Stakeholder | Value Delivered |
|---|---|
| Political campaigns | Identify high-ROI ad timing and spend thresholds to maximize reach without guessing |
| Platform policy teams | Auditable framework to demonstrate algorithmic fairness to regulators |
| Compliance & legal | Reproducible statistical evidence that engagement models don't discriminate by ad group |
| Data science teams | End-to-end template for missingness-aware modeling + fairness evaluation in a single pipeline |

This analysis directly supports the kind of transparency reporting now required by the EU Digital
Services Act and emerging U.S. political ad disclosure frameworks.

---

## Skills & Tech Stack

| Category | Tools |
|---|---|
| Data wrangling | Pandas, NumPy |
| Statistical testing | Permutation tests, TVD, custom hypothesis testing pipelines |
| Feature engineering | pytz, address parsing, time zone normalization |
| Modeling | Scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, GridSearchCV) |
| Fairness analysis | Custom FPR parity evaluation via permutation testing |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3 |

___

> **Note:** This project was completed as part of a formal data science curriculum. All analysis uses publicly available data.
