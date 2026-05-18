<div align="center">

# 🏠 Loan Default Prediction

### Regularisation, Feature Importance, and Fairness on the Home Credit Default Risk Dataset

[![Built with](https://img.shields.io/badge/Built_with-Python_%7C_Jupyter-ffe05a?logo=python&logoColor=white)]()
[![Models](https://img.shields.io/badge/Models-LogReg_%7C_ElasticNet_%7C_RF_%7C_XGBoost_%7C_MLP-ee901e)]()
[![Data](https://img.shields.io/badge/Data-Home_Credit_Default_Risk-22beff)]()

*A consolidated notebook that predicts whether an applicant will default on a loan, comparing five model families on the same 70/15/15 split and using regularisation as the lens to study predictive performance, feature importance, and subgroup fairness.*

</div>

#

<br>

## Overview

This project predicts loan default on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset, a 307,511 applicant table described by 122 features covering income, employment, housing, and credit history. The target is binary (1 = early-installment payment difficulties, 0 = otherwise) and heavily imbalanced, with only 8.07% positives (24,825 defaulters).

The analysis is structured around three research questions:

* **RQ1.** How do different regularisation strategies (L1, L2, Elastic Net, dropout, batch normalisation, early stopping) affect predictive performance and model complexity?
* **RQ2.** Which applicant features are most predictive of default, and do the most important features differ across model families (linear, tree-based, neural network)?
* **RQ3.** Do demographic attributes (gender, age, income) appear as predictive features across model families, and what does the way different architectures weight them imply for fairness in credit scoring?

Five models are trained and evaluated on the same splits to keep the comparison fair: an L2 **logistic regression baseline**, a tuned **elastic net logistic regression**, a tuned **random forest**, a tuned **XGBoost** classifier, and a custom **Keras MLP**. Linear and neural variants are also run on a 49 component PCA branch, and every classical model is benchmarked against a SMOTE oversampled training set.

```
307,511 applications, 122 raw features, 8.07% positive class
Stratified 70 / 15 / 15 split, 215,253 / 46,126 / 46,126 (6 rows dropped on unknowns)
Default rate preserved across all three partitions
Fixed seed = 42
```

<br>

## Results

**Headline model: XGBoost** wins on both ROC-AUC and PR-AUC on the held-out test set. The MLP comes in second; Elastic Net, the LR baseline, and Random Forest are statistically tied behind it.

### Cross-model test performance

| Model | Test ROC-AUC | Test PR-AUC |
|-------|-------------:|------------:|
| XGBoost (tuned)        | **0.7689** | **0.2585** |
| MLP (Keras, best config)| 0.7571 | 0.2415 |
| Elastic Net (tuned)    | 0.7523 | 0.2350 |
| LR Baseline (L2)       | 0.7521 | 0.2350 |
| Random Forest (tuned)  | 0.7517 | 0.2287 |

### Key findings

* **Regularisation does two different jobs (RQ1).** For the linear family the 20 cell Elastic Net grid sits inside a 0.0003 cross-validated AUC spread (0.7482 to 0.7485), yet drives coefficient sparsity from 0% to 34.5%. Regularisation buys parsimony, not accuracy, because the linear models are already at the generalisation ceiling set by the EXT_SOURCE block and the engineered ratios. For the MLP, combining Dropout, BatchNorm, and Early Stopping closes the train/val AUC gap from 0.1475 to 0.0041, two orders of magnitude smaller, while pushing val AUC to its highest of 0.7580. Random Forest and XGBoost show the same pattern (train/val gaps of 0.088 and 0.061 in the tuned fits).
* **Feature importance disagrees sharply across families (RQ2).** Within the linear family, Elastic Net and the LR baseline agree at Spearman ρ = 0.917, with 8 of 10 features shared between the top-10 lists. Across families that agreement collapses to ρ ≈ 0.13 to 0.15 (LR vs Random Forest Gini and LR vs Random Forest Permutation). Even within XGBoost, Gain versus Permutation importance only reaches ρ = 0.513. Linear models surface monetary amounts, missingness flags, and one-hot income categories; tree models concentrate importance on continuous variables, especially EXT_SOURCE_2 and EXT_SOURCE_3.
* **Demographic attributes surface in every top ranking (RQ3).** Male applicants default at 10% against 7% for female applicants, and NAME_INCOME_TYPE spans 40% (maternity leave) and 36% (unemployed) against 5% (pensioners) before any model is fit. These base-rate gaps then surface in the models: CODE_GENDER_M ranks 8th in the Elastic Net coefficients and 3rd in XGBoost Gain, and NAME_INCOME_TYPE_Pensioner is the single largest Elastic Net coefficient.
* **SMOTE underperformed class_weight on every classical model.** Test ROC-AUC drops with SMOTE for LR (-0.007), Elastic Net (-0.007), Random Forest (-0.033), and XGBoost (-0.013), in line with the van den Goorbergh et al. (2022) result that SMOTE overestimates minority class probability without improving AUC.
* **PCA does not help.** The 95% variance threshold cuts 113 features to 49 components, but PCA variants underperform their original feature counterparts for every model family.

<br>

## Methodology

1. **Data loading.** The cleaned `application_train.csv` is hosted on Google Drive and pulled automatically via `gdown` on the first run. The auxiliary Home Credit tables (bureau, previous applications, installment payments) are deliberately left out; the main table alone provides 307,511 rows and enough features to answer the three research questions without inflating the pipeline.

2. **EDA.** Five passes over the training data: univariate distributions for every feature, bivariate default rate by demographic and contract categoricals, Pearson correlation on the financial columns, IQR based outlier scan, and a missing values audit that groups 67 null carrying columns into six reason based families.

3. **Pre-processing.** Six rows dropped on unknown gender or family status, leaving 307,505 rows. The DAYS_EMPLOYED sentinel (365243) is replaced with NaN and surfaced as a DAYS_EMPLOYED_ANOM flag. Four ratio features engineered (CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM, CREDIT_GOODS_RATIO). 66 columns dropped (the G1 apartment / building appraisal block at 47% to 70% missing, near-duplicate pairs such as REGION_RATING_CLIENT_W_CITY at r = 0.95, zero-variance flags, rare FLAG_DOCUMENT_* indicators, OWN_CAR_AGE, and the ID columns). The four monetary columns are winsorised at the 99th percentile, log1p transformed, then standardised. EXT_SOURCE_1, _2, _3 get median imputation paired with an EXT_SOURCE*_MISSING flag so absence itself becomes a signal. NAME_EDUCATION_TYPE is ordinal encoded; the other 11 retained categoricals are one-hot encoded. The standard pipeline ends at 113 features; a parallel PCA branch retains 49 components for 95% variance. SMOTE is applied to the training set only (215,253 to 395,752 rows); validation and test are left at the natural 8.07% default rate.

4. **Modelling.** The LR baseline uses L2 with `class_weight='balanced'`, default C = 1.0, and the lbfgs solver. Elastic Net is fit with the saga solver across a `GridSearchCV` over C ∈ {0.01, 0.10, 1.0, 10.0} and l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, 20 configurations on 5 stratified folds. Random Forest is tuned via `RandomizedSearchCV` over n_estimators, max_depth, max_leaf_nodes, and min_samples_split (15 configs, 5 folds). XGBoost uses `scale_pos_weight` set to the negative-to-positive ratio and a 20 configuration `RandomizedSearchCV` over six hyperparameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight). The MLP is a three-layer feedforward Keras network with Leaky ReLU activations (α = 0.1), Dropout 0.5, BatchNorm, class weighted binary cross-entropy, Adam (lr = 0.001), and Early Stopping on val AUC; 12 sweep configurations identify the best architecture (23,489 parameters). Each model gets a SMOTE counterpart, and linear plus MLP variants are additionally fit on the 49 component PCA inputs.

5. **Evaluation.** ROC-AUC and PR-AUC are the primary numbers since both are threshold independent and PR-AUC is the more honest summary under 8.07% prevalence. Each model gets its own F1 optimal threshold, swept on the validation set, and that operating point is scored exactly once on the held-out test set. Feature importance is read two ways for the tree models (Gini / Gain plus permutation) and as standardised coefficients for the linear models. A five variant MLP regularisation ablation (no reg, Dropout only, BN only, ES only, all three combined) isolates each regulariser's contribution to the train/val gap.

6. **Reproducibility.** A single `SEED = 42` is fixed across NumPy, TensorFlow, and every sklearn `random_state`. Expensive fits and CV searches are checkpointed to `checkpoints/` so re-runs only redo what changed.

<br>

## Data Source

[**Home Credit Default Risk**](https://www.kaggle.com/c/home-credit-default-risk), Kaggle competition data released by Home Credit Group (2018).

The notebook downloads `application_train.csv` automatically from a public Google Drive link on the first run. No Kaggle API key, no manual download, no setup beyond `pip install`. The `data/Data.7z` archive in this repository is a compressed backup of the same file, kept only as a fallback in case the Google Drive link becomes unavailable; the notebook does not read from it.

> The Kaggle competition's `application_test.csv` has no TARGET column (labels were never released after the competition closed), so the train / validation / test split is carved out of `application_train.csv` directly. Every per-column decision in preprocessing follows `HomeCredit_columns_description.csv`, the data dictionary that ships with the dataset.

<br>

## Repository Structure

```
loan-default-prediction/
│
├── MLDL_FinalExamCode_Group30.ipynb   # Consolidated notebook, runs end to end top to bottom
│
├── data/
│   └── Data.7z                        # Backup of application_train.csv (not used by the notebook)
│
├── .gitignore
└── README.md
```

> **Note.** `data/`, `checkpoints/`, and `outputs/` are created next to the notebook on the first run. `data/application_train.csv` is downloaded automatically from Google Drive, model checkpoints and the best MLP land in `checkpoints/`, and every generated plot is saved to `outputs/` with an auto-incremented numeric prefix.

<br>

## Technologies Used

* **Python 3** in **Jupyter**
* [`pandas`](https://pandas.pydata.org/) and [`numpy`](https://numpy.org/) for data handling, [`matplotlib`](https://matplotlib.org/) and [`seaborn`](https://seaborn.pydata.org/) for plotting.
* [`scikit-learn`](https://scikit-learn.org/) for the `ColumnTransformer` preprocessing pipeline, logistic regression, elastic net, random forest, PCA, `GridSearchCV` / `RandomizedSearchCV`, and the full evaluation toolkit (ROC-AUC, PR-AUC, confusion matrix, ROC and PR curves, permutation importance).
* [`imbalanced-learn`](https://imbalanced-learn.org/) for SMOTE oversampling on the 8.07% positive class.
* [`xgboost`](https://xgboost.readthedocs.io/) for the gradient-boosted-tree model with `scale_pos_weight`.
* [`tensorflow` / `keras`](https://www.tensorflow.org/) for the custom MLP with Leaky ReLU, dropout, batch normalisation, and early stopping.
* [`gdown`](https://github.com/wkentaro/gdown) to pull the dataset from Google Drive on the first run, [`joblib`](https://joblib.readthedocs.io/) for checkpointing fitted models and search results, [`scipy`](https://scipy.org/) for the Spearman rank correlations used in the RQ2 cross-family comparison.

<br>

## How to Run

1. Clone the repository.
   ```bash
   git clone https://github.com/zakyikow/loan-default-prediction.git
   cd loan-default-prediction
   ```
2. Install dependencies.
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow gdown joblib scipy
   ```
3. Open `MLDL_FinalExamCode_Group30.ipynb` and run all cells top to bottom.

The notebook downloads the data automatically, creates `data/`, `checkpoints/`, and `outputs/` next to itself on the first run, and checkpoints expensive steps so re-runs only redo what changed. No manual download, no API key, no setup beyond `pip install`.

> **Heads up on runtimes.** Final fits are cheap (LR baseline in seconds, XGBoost and MLP in tens of seconds, tuned Elastic Net and Random Forest in a few minutes). The hyperparameter searches are the expensive part: about 30 minutes for XGBoost, 2.7 hours for Random Forest, and almost 7 hours for the Elastic Net `GridSearchCV` on a MacBook Pro M1. Joblib checkpoints make repeat runs essentially free.

<br>

## Known Limitations

1. **Single-table modelling.** The Home Credit auxiliary tables (bureau, previous applications, installment payments, etc.) are not joined in. Top public Kaggle submissions clear 0.80 ROC-AUC precisely because they aggregate that extra signal, so the 0.7689 test ROC-AUC ceiling partly reflects leaving those tables aside.
2. **No competition leaderboard score.** Because `application_test.csv` has no released labels, the test set is an in sample split, not the Kaggle hold-out. Numbers are not directly comparable to the public Kaggle leaderboard.
3. **Narrow Elastic Net grid.** All four C values produced the same flat 0.0003 AUC band, so the grid did not locate the regularisation strength at which AUC starts to drop. C values of 1e-4 and 1e-3 would have been needed.
4. **No formal fairness audit.** RQ3 stays at the level of feature-importance rankings and base rates. A real deployment would need per-subgroup ROC-AUC, calibration, and precision / recall at the operating threshold, plus tests for whether regularisation or resampling closes any observed gaps.
5. **No variance bars.** Each configuration was trained once, so the gap between the three statistically tied classical models (Elastic Net, LR baseline, Random Forest) sits within plausible seed-level variability and should not be over-interpreted.
