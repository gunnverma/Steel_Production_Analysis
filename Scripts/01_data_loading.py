
import pandas as pd

print("starting data loading...")

# reading the csv files
train_data = pd.read_csv("../data/normalized_train_data.csv")
test_data = pd.read_csv("../data/normalized_test_data.csv")

# checking if it loaded ok
print("Training data shape:")
print(train_data.shape)
print(train_data.head())

print("Test data shape:")
print(test_data.shape)
print(test_data.head())

print("Data loaded successfully")





