import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pyreadr
import os
# --- Prepare RDS Output Using rpy2 ---
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()



# --- Reading and Preparing Data ---
result = pyreadr.read_r("ALS_progression_rate.1822x370.rds")
df = result[None]  # Extract the data frame from the RDS file
df = df.rename(columns={'dFRS': 'response'})

data_train = df[df['response'].notna()]
data_predict = df[df['response'].isna()]

print(f"Training dim: {data_train.shape}")
print(f"Prediction dim: {data_predict.shape}")

X = data_train.drop('response', axis=1)
y = data_train['response']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Fitting ---
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")

y_pred = lasso_cv.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, y_val, alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.title("Observed vs Predicted")

residuals = y_val - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

# --- Prediction for Unlabeled Data ---
X_predict = data_predict.drop('response', axis=1)
predictions = lasso_cv.predict(X_predict)
submission = pd.DataFrame({'predicted': predictions})


os.environ["R_HOME"] = "C:/Program Files/R/R-4.4.2"  

team_name = "team_ashkill"
team_people = ["Shakil", "Ashrif"]
team_error_rate = rmse
team_predictions = submission[['predicted']]  

r_team_people = ro.StrVector(team_people)
r_team_predictions = pandas2ri.py2rpy(team_predictions)
r_list = ro.r['list'](team_name, r_team_people, team_error_rate, r_team_predictions)
output_filename = f"als_progression.{team_name}.rds"


ro.r['saveRDS'](r_list, file=output_filename)
print(f"RDS file saved as {output_filename}")
