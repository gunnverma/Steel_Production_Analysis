
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

print("Starting Results Analysis...")

# loading metrics
results = pd.read_csv("../results/performance_metrics.csv")

# 1. Bar Plot for RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results)
plt.title("RMSE Comparison")
plt.savefig("../figures/model_comparison.png")
print("Saved model_comparison.png")

# 2. Prediction vs Actual Scatter Plot 
# loading the prediction files 
rf_preds = pd.read_csv("../results/model_predictions/Random_Forest.csv")
svr_preds = pd.read_csv("../results/model_predictions/SVR.csv")
mlp_preds = pd.read_csv("../results/model_predictions/MLP.csv")
gp_preds = pd.read_csv("../results/model_predictions/Gaussian_Process.csv")

plt.figure(figsize=(12, 12))

# Plot 1: Random Forest
plt.subplot(2, 2, 1)
plt.scatter(rf_preds['Actual'], rf_preds['Predicted'], alpha=0.3, color='blue')
plt.plot([0,1], [0,1], 'r--')
plt.title("Random Forest")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Plot 2: SVR
plt.subplot(2, 2, 2)
plt.scatter(svr_preds['Actual'], svr_preds['Predicted'], alpha=0.3, color='pink')
plt.plot([0,1], [0,1], 'r--')
plt.title("SVR")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Plot 3: MLP
plt.subplot(2, 2, 3)
plt.scatter(mlp_preds['Actual'], mlp_preds['Predicted'], alpha=0.3, color='green')
plt.plot([0,1], [0,1], 'r--')
plt.title("MLP (Neural Network)")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Plot 4: Gaussian Process
plt.subplot(2, 2, 4)
plt.scatter(gp_preds['Actual'], gp_preds['Predicted'], alpha=0.3, color='purple')
plt.plot([0,1], [0,1], 'r--')
plt.title("Gaussian Process")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig("../figures/predictions_vs_actual.png")
print("Saved predictions_vs_actual.png")


# 3. Residual Plots (Errors)

# calculating residuals (Actual - Predicted)
rf_resid = rf_preds['Actual'] - rf_preds['Predicted']
svr_resid = svr_preds['Actual'] - svr_preds['Predicted']
mlp_resid = mlp_preds['Actual'] - mlp_preds['Predicted']
gp_resid = gp_preds['Actual'] - gp_preds['Predicted']

plt.figure(figsize=(12, 12))

# Random Forest Residuals
plt.subplot(2, 2, 1)
plt.scatter(rf_preds['Predicted'], rf_resid, alpha=0.3, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title("Random Forest Residuals")
plt.xlabel("Predicted")
plt.ylabel("Error")

# SVR Residuals
plt.subplot(2, 2, 2)
plt.scatter(svr_preds['Predicted'], svr_resid, alpha=0.3, color='red')
plt.axhline(0, color='red', linestyle='--')
plt.title("SVR Residuals")
plt.xlabel("Predicted")
plt.ylabel("Error")

# MLP Residuals
plt.subplot(2, 2, 3)
plt.scatter(mlp_preds['Predicted'], mlp_resid, alpha=0.3, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title("MLP Residuals")
plt.xlabel("Predicted")
plt.ylabel("Error")

# Gaussian Process Residuals
plt.subplot(2, 2, 4)
plt.scatter(gp_preds['Predicted'], gp_resid, alpha=0.3, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Gaussian Process Residuals")
plt.xlabel("Predicted")
plt.ylabel("Error")

plt.tight_layout()
plt.savefig("../figures/residuals.png")
print("Saved residuals.png")


# 4. Learning Curve (Random Forest)

print("Generating Learning Curve...")

# loading data again just for this plot
train = pd.read_csv("../data/clean_train.csv")
X = train.drop('output', axis=1)
y = train['output']

# using sklearn to calculate the curve data
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestRegressor(n_estimators=50, random_state=42),
    X, y,
    cv=3,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

# converting scores to positive RMSE
train_mean = np.sqrt(-train_scores.mean(axis=1))
test_mean = np.sqrt(-test_scores.mean(axis=1))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Error")
plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Error")
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Training Examples")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.savefig("../figures/learning_curve.png")
print("Saved learning_curve.png")

print("All Analysis Complete!")

