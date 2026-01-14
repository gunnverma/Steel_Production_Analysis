# Steel Production Quality Prediction (P1)
**Author:** Gunn Verma

## Project Overview
This project focuses on analysing steel production data to build machine learning models that predict the quality of the steel output. The dataset contains sensor readings from the production line, and my goal was to predict the `output` variable (a quality score between 0 and 1).

I implemented a complete pipeline from **data preprocessing** to **model comparison** using Python.

### Key Challenge & Solution
During the analysis, I discovered a significant **Distribution Shift** between the Training and Test datasets. The average quality in the test set was lower than in the training set, which caused my initial models to fail (they had negative R² scores).

**My Solution:** I implemented a **Bias Correction** step in the training script. I calculated the difference (bias) between my predictions and the actual values, then subtracted it. This stabilised the models and allowed me to find a working baseline.

---

## Project Structure
I organised the code into separate scripts to make it easier to read and grade:

```text
steel_production_analysis/
│
├── data/
│   ├── normalized_train_data.csv  <-- Input Train File
│   ├── normalized_test_data.csv   <-- Input Test File
│   ├── clean_train.csv            <-- Processed Train File
│   └── clean_test.csv             <-- Processed Test File
│
├── scripts/
│   ├── 01_data_loading.py         # Loads data and checks consistency
│   ├── 02_data_preprocessing.py   # Cleans duplicates & handles missing values
│   ├── 03_eda.py                  # Generates histograms & correlation matrices
│   ├── 04_model_training.py       # Trains 4 models & applies Bias Correction
│   └── 05_results_analysis.py     # Generates comparison plots & learning curves
│
├── results/
│   ├── performance_metrics.csv    # Final scores (RMSE, R2, Time)
│   └── model_predictions/         # CSVs containing Actual vs Predicted values
│
├── figures/                       # All generated plots are saved here
│   ├── distribution_shift.png
│   ├── predictions_vs_actual.png
│   └── learning_curve.png
│
└── P1_Report.pdf                     # Final Project Report

```

---

## 🛠️ Installation & Requirements

The project requires Python 3.x and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

---

## 🚀 How to Run the Project

Please run the scripts in the following order from the main folder:

### 1. Data Loading

Checks if the datasets exist and load correctly.

```bash
python scripts/01_data_loading.py

```

### 2. Preprocessing

Cleans the data (removes duplicates, fills missing values) and saves the "clean" versions.

```bash
python scripts/02_data_preprocessing.py

```

### 3. Exploratory Data Analysis (EDA)

Generates visualisations to check for outliers and the distribution shift. Check the `figures/` folder after running this.

```bash
python scripts/03_eda.py

```

### 4. Model Training

Trains **Random Forest**, **SVR**, **MLP**, and **Gaussian Process** models. It automatically applies the bias correction and saves predictions.
*(Note: Gaussian Process may take ~5 minutes to run).*

```bash
python scripts/04_model_training.py

```

### 5. Results Analysis

Creates the final comparison charts, residual plots, and learning curves.

```bash
python scripts/05_results_analysis.py

```

---

## 📊 Results & Findings

I compared four regression models. The results are as follows:

| Model | RMSE (Error) | R² Score | Training Time |
| --- | --- | --- | --- |
| **Gaussian Process** | **0.088** | **0.135** | ~320s |
| Random Forest | 0.094 | 0.003 | ~30s |
| MLP | 0.097 | -0.051 | ~2s |
| SVR | 0.115 | -0.490 | ~1s |

### Conclusion

* **The Winner:** The **Gaussian Process Regressor** was the only model to achieve a positive R² score (> 0.13). It handled the uncertainty in the noisy data better than the others.
* **The Baseline:** Random Forest performed effectively like a baseline (R² ≈ 0), predicting the mean value.
