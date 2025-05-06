from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
# Generate dummy dataset
X = np.random.rand(100, 5) # 100 samples, 5 features
y = np.random.randint(0, 2, 100) # Binary target variable (0 or 1)
# Splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create a Naïve Bayes model
model = GaussianNB()
# Train the model on training data
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")



# import pandas as pd
# df = pd.read_csv("data.csv") # Reading a CSV file
# df.to_csv("output.csv", index=False) # Writing to a CSV file
# print(df.head()) # Display first few rows


# 3.2 Handling Missing Values
# Missing values can cause issues in model training. We can handle them in multiple ways:
# # Check for missing values
# print(df.isnull().sum())

# # Drop missing values
# df_cleaned = df.dropna()
# print("Dataset after dropping missing values:", df_cleaned.shape)

# # Handle missing values
# df['age'] = df['age'].fillna(df['age'].mean()) # Fill age with mean
# df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0]) # Fill
# embarked with mode

# # Convert deck to string and replace NaN
# df['deck'] = df['deck'].astype(str).fillna('Unknown')
# # Drop remaining NaN values if needed
# df = df.dropna()
# print(df.isnull().sum()) # Confirm no missing values



# 3.3 Encoding Categorical Data
# Machine learning models require numerical data, so we must convert categorical variables.

# 3.3.1 Label Encoding (For Binary Categories)
# #Converts categories into numbers (e.g., male → 0, female → 1).
# le = LabelEncoder()
# df['gender'] = le.fit_transform(df['gender']) # 0 for male, 1 for female
# print(df['gender'].head())

# 3.3.2 One-Hot Encoding (For Multi-Class Categories)
# #Creates separate binary columns for each category.
# df = pd.get_dummies(df, columns=['embarked'], drop_first=True)
# print(df.head())


# 3.4 Data Visualization
# Data visualization is the process of representing data graphically using plots, charts, and
# graphs. It helps to identify patterns and trends in the data.

# #creating histogram of Numerical Features
# df[['age', 'fare']].hist(bins=20, figsize=(8,4))
# plt.show()