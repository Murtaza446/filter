import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = sns.load_dataset("titanic")

# Select features and target, handling missing values
X = df[['age', 'fare']].fillna(df[['age', 'fare']].mean())
y = df['survived']

# Convert to DataFrame and Series
X = pd.DataFrame(X)
y = pd.Series(y)

# Initialize model
model = LogisticRegression()

# K-Fold Cross-Validation
def perform_kfold_cv(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print("K-Fold CV Average Accuracy:", np.mean(accuracy_scores))

# Leave-One-Out Cross-Validation
def perform_loocv(X, y, model):
    loo = LeaveOneOut()
    loo_scores = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loo_scores.append(accuracy_score(y_test, y_pred))
    print("LOOCV Average Accuracy:", np.mean(loo_scores))

# Perform K-Fold CV
perform_kfold_cv(X, y, model)

# Perform LOOCV
perform_loocv(X, y, model)