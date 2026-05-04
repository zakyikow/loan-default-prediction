# Loan Default Prediction, Regularization as a Lens

## Team

[names and student IDs]

## About

we predict whether an applicant will default on a loan using the home credit default risk dataset. the analysis is structured around three research questions, the effect of regularisation on each model (RQ1), how feature importance compares across models (RQ2), and fairness across demographic subgroups (RQ3). we compare a logistic regression baseline, an elastic net, a random forest, an XGBoost model, and a custom MLP, all evaluated on the same train, val, and test splits.

course, machine learning and deep learning (KAN-CDSCO2004U), CBS.

## How to run

1. Clone this repo.

   ```
   git clone <repo-url>
   cd <repo-name>
   ```

2. Install dependencies.

   ```
   pip install -r requirements.txt
   ```

3. Open `notebook.ipynb` and run all cells top to bottom. the notebook downloads the data automatically from google drive, creates `data/`, `checkpoints/`, and `outputs/` folders as it runs, and checkpoints expensive steps so re-runs only redo what changed. there is no manual download, no API key, no setup beyond the pip install.

## Dataset

home credit default risk, hosted on kaggle at https://www.kaggle.com/datasets/diogoamaro0/loan-deafult-prediction
