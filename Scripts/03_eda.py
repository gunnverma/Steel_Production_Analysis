
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the clean data
train = pd.read_csv("../data/clean_train.csv")
test = pd.read_csv("../data/clean_test.csv")

# checking the target distribution (Shift check)
# this is important because the mean is different
plt.figure(figsize=(10, 6))
sns.kdeplot(train['output'], fill=True, color='blue', label='Train')
sns.kdeplot(test['output'], fill=True, color='red', label='Test')
plt.title("Shift in Output Variable")
plt.legend()
plt.savefig("../figures/distribution_shift.png")
print("made distribution plot")

# correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), cmap="coolwarm")
plt.title("Correlation")
plt.savefig("../figures/correlation_matrix.png")
print("made correlation plot")

# histograms for first few columns
# only doing for the first 4 to visualize
plt.figure(figsize=(12, 8))

# Plot 1
plt.subplot(2, 2, 1)
sns.histplot(train['input1'], color='pink')
plt.title("Input 1 Distribution")

# Plot 2
plt.subplot(2, 2, 2)
sns.histplot(train['input2'], color='orange')
plt.title("Input 2 Distribution")

# Plot 3
plt.subplot(2, 2, 3)
sns.histplot(train['input3'], color='green')
plt.title("Input 3 Distribution")

# Plot 4
plt.subplot(2, 2, 4)
sns.histplot(train['input4'], color='purple')
plt.title("Input 4 Distribution")

plt.tight_layout()
plt.savefig("../figures/histograms.png")
print("made histograms")



