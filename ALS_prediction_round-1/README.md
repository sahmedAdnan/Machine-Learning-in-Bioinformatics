# ALS Progression Prediction using LASSO Regression

This repository contains a Python-based machine learning pipeline for predicting the progression rate of ALS (Amyotrophic Lateral Sclerosis) using LASSO regression with cross-validation. The pipeline reads data from an RDS file, trains a model, evaluates its performance, and generates predictions for unlabeled data. Finally, it exports the results (including team metadata) into an RDS file that can be processed in R.


- **lasso_regression.py**: The Python script implementing the LASSO regression pipeline.

## Prerequisites

- **Python 3.x**
- **R installation** (required for saving the output RDS file)
- Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `pyreadr`
  - `rpy2`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/sahmedAdnan/Machine-Learning-in-Bioinformatics.git
   cd ALS_prediction_round-1
   ```
2. **Create and Activate a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv env
   or
   Conda activate <myenv>
   ```
3. **Set the `R_HOME` Environment Variable:**
- **Windows Example:**
```bash
set R_HOME="C:\Program Files\R\R-4.4.2"
```
- **Linux/Mac Example:**
```bash
export R_HOME="/usr/lib/R"
```

# ALS Progression Prediction

## Usage

1. **Data Preparation:**
   * Place your input RDS file (`ALS_progression_rate.1822x370.rds`) in the repository root.

2. **Run the Script:**
   ```bash
   python lasso_regression.py
   ```

3. **Pipeline Workflow:**
   * **Data Loading:** Reads the ALS progression data from an RDS file and renames the target column from `dFRS` to `response`.
   * **Data Splitting:** Divides the data into training data (with a response) and unlabeled data (without a response).
   * **Model Training:** Splits the training data further into training and validation sets and trains a LASSO regression model with 5-fold cross-validation.
   * **Model Evaluation:** Computes the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the validation set and visualizes predictions versus observed values and residuals.
   * **Prediction:** Uses the trained model to predict the progression for unlabeled data.
   * **Output:** Saves the predictions along with team metadata into an RDS file (`als_progression.team_ashkill.rds`) using `rpy2` so that it can be read in R.

## Code Overview

The main script (`lasso_regression.py`) is structured as follows:

* **Library Imports:** Loads necessary Python libraries for data manipulation, model training, evaluation, and R integration.
* **Data Reading and Preparation:** Reads the input data from an RDS file and renames columns.
* **Data Splitting:** Divides the data into training (with a known response) and prediction (unlabeled) sets. Further splits the training data for validation.
* **LASSO Regression Model:** Utilizes `LassoCV` from scikit-learn to select the best regularization parameter (`alpha`) using cross-validation. Evaluates model performance using RMSE.
* **Visualization:** Plots observed versus predicted values and residuals to help assess model performance.
* **Prediction on Unlabeled Data:** Applies the model to predict ALS progression for data without a response.
* **Export to RDS:** Uses `rpy2` to convert Python objects to R-compatible formats and saves them as an RDS file.

## Customization

* **Team Details:** Update the `team_name` and `team_people` variables in the script to reflect your team's information.
* **Model Parameters:** Adjust the LASSO parameters (such as `max_iter` or cross-validation settings) to fine-tune model performance.

## Troubleshooting

* **R_HOME Error:** Ensure that the `R_HOME` environment variable is correctly set to the path of your R installation.
* **Package Issues:** Confirm that all dependencies listed in are installed.






