import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file

# 1. Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)  # For numeric columns
df.fillna(df.mode().iloc[0], inplace=True)  # For categorical columns

# 2. Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Standardize numerical features
scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('target')  # change target
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 4. Split features and target
X = df.drop('target', axis=1)  # Replace 'target' with your actual target column
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

# Evaluation on testing data
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Accuracy: ", cv_scores.mean())