import pandas as pd
from pandas import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('data.csv')
df.head()

print(df.isnull().sum())

df['column_name'] = df['column_name'].fillna(df['column_name'].mean(), inplace=True)
# or df['column_name'].fillna(df['column_name'].mode()[0], inplace=True) for categorical data
# or df['column_name'].fillna(df['column_name'].median(), inplace=True) for numerical data
#df['column_name'] = df['column_name'].astype(str).fillna('missing')
#df.dropna()
df.dropna(subset=['column_name'], inplace=True)

le = LabelEncoder()
df['column_name'] = le.fit_transform(df['column_name'])#for binary categorical data
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

#df['age'] = scaler.fit_transform(df[['age']])
#df['fare'] = scaler.fit_transform(df[['fare']])

# Select features and target, handling missing values
X = df[['age', 'fare']].fillna(df[['age', 'fare']].mean())
y = df['survived']

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
accuracy_score1 = accuracy_score(y_test, y_pred1) 

model2 = GaussianNB()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
# mse2 = mean_squared_error(y_test, y_pred2)
accuracy_score2 = accuracy_score(y_test, y_pred2)

model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
training_accuracy = model3.score(X_train, y_train)
testing_accuracy = model3.score(y_test, y_pred3)
accuracy_score3 = accuracy_score(y_test, y_pred3)




from sklearn.metrics import roc_curve, roc_auc_score
# Get probabilities for the positive class
probabilities = model3.predict_proba(X_test)[:, 1] # Get the probability for class
'1'
# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
# Calculate ROC_AUC Score
roc_auc = roc_auc_score(y_test, probabilities)
# Plot ROC curve with shaded area under the curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC ={roc_auc:.2f})')
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.4)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with AUC Area')
plt.legend(loc='lower right')
plt.show()

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

X = pd.DataFrame(X)
y = pd.Series(y)
# Define K-Fold (5 splits)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize model
model = LogisticRegression()
# Store accuracy scores
accuracy_scores = []
# Perform K-Fold CV
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # Now X is aDataFrame
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Now y is aSeries
    # Train model
    model.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
# Print average accuracy
print("K-Fold CV Average Accuracy:", np.mean(accuracy_scores))



