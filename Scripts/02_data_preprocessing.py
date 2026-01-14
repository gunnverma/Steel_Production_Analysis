
import pandas as pd
import numpy as np


print("Processing Train Data...")
df_train = pd.read_csv("../data/normalized_train_data.csv")

# removing duplicates
old_len = len(df_train)
df_train = df_train.drop_duplicates()
new_len = len(df_train)
print("Duplicates removed in train:", old_len - new_len)

# handling missing values
# filling with 0 because data is normalized
df_train = df_train.fillna(0)

# detecting outliers with IQR
# also printing how many there are
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = ((df_train < lower) | (df_train > upper)).sum().sum()
print("Total outliers found in train:", outliers)

# saving clean train file
df_train.to_csv("../data/clean_train.csv", index=False)
print("Saved clean_train.csv")


# (I just copied the code from above for the test set)
print("Processing Test Data...")
df_test = pd.read_csv("../data/normalized_test_data.csv")

# removing duplicates
df_test = df_test.drop_duplicates()

# filling nan
df_test = df_test.fillna(0)

# saving clean test file
df_test.to_csv("../data/clean_test.csv", index=False)
print("Saved clean_test.csv")



