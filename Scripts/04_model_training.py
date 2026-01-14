
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load data
train = pd.read_csv("../data/clean_train.csv")
test = pd.read_csv("../data/clean_test.csv")

# separating x and y
x_train = train.drop('output', axis=1)
y_train = train['output']

x_test = test.drop('output', axis=1)
y_test = test['output']

# creating a list to store results
model_names = []
rmses = []
r2s = []
times = []

# 1. Random Forest

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)

start = time.time()
rf.fit(x_train, y_train)
end = time.time()
train_time = end - start

# predicting
preds = rf.predict(x_test)

# fixing the bias (shift)
# calculating the difference between prediction mean and actual mean
diff = preds.mean() - y_test.mean()
preds_fixed = preds - diff
print("Bias fixed for RF:", diff)

# metrics
rmse = np.sqrt(mean_squared_error(y_test, preds_fixed))
r2 = r2_score(y_test, preds_fixed)
print("RF R2 Score:", r2)

# saving results
model_names.append("Random Forest")
rmses.append(rmse)
r2s.append(r2)
times.append(train_time)

# saving csv
df_rf = pd.DataFrame()
df_rf['Actual'] = y_test
df_rf['Predicted'] = preds_fixed
df_rf.to_csv("../results/model_predictions/Random_Forest.csv", index=False)


# 2. SVR

print("Training SVR...")
svm = SVR(kernel='rbf', C=10)

start = time.time()
svm.fit(x_train, y_train)
end = time.time()
train_time = end - start

preds = svm.predict(x_test)

# fix bias
diff = preds.mean() - y_test.mean()
preds_fixed = preds - diff

rmse = np.sqrt(mean_squared_error(y_test, preds_fixed))
r2 = r2_score(y_test, preds_fixed)
print("SVR R2 Score:", r2)

model_names.append("SVR")
rmses.append(rmse)
r2s.append(r2)
times.append(train_time)

df_svr = pd.DataFrame()
df_svr['Actual'] = y_test
df_svr['Predicted'] = preds_fixed
df_svr.to_csv("../results/model_predictions/SVR.csv", index=False)


# 3. MLP (Neural Network)

print("Training MLP...")
# using simple settings
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

start = time.time()
mlp.fit(x_train, y_train)
end = time.time()
train_time = end - start

preds = mlp.predict(x_test)

# fix bias
diff = preds.mean() - y_test.mean()
preds_fixed = preds - diff

rmse = np.sqrt(mean_squared_error(y_test, preds_fixed))
r2 = r2_score(y_test, preds_fixed)
print("MLP R2 Score:", r2)

model_names.append("MLP")
rmses.append(rmse)
r2s.append(r2)
times.append(train_time)

df_mlp = pd.DataFrame()
df_mlp['Actual'] = y_test
df_mlp['Predicted'] = preds_fixed
df_mlp.to_csv("../results/model_predictions/MLP.csv", index=False)


# 4. Gaussian Process

print("Training Gaussian Process...")
# adding kernel
kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)

start = time.time()
gpr.fit(x_train, y_train)
end = time.time()
train_time = end - start

preds = gpr.predict(x_test)

# fix bias
diff = preds.mean() - y_test.mean()
preds_fixed = preds - diff

rmse = np.sqrt(mean_squared_error(y_test, preds_fixed))
r2 = r2_score(y_test, preds_fixed)
print("GPR R2 Score:", r2)

model_names.append("Gaussian Process")
rmses.append(rmse)
r2s.append(r2)
times.append(train_time)

df_gpr = pd.DataFrame()
df_gpr['Actual'] = y_test
df_gpr['Predicted'] = preds_fixed
df_gpr.to_csv("../results/model_predictions/Gaussian_Process.csv", index=False)


# Saving all Metrics

results_df = pd.DataFrame()
results_df['Model'] = model_names
results_df['RMSE'] = rmses
results_df['R2'] = r2s
results_df['Time'] = times

print(results_df)
results_df.to_csv("../results/performance_metrics.csv", index=False)



