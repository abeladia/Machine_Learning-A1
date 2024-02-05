import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load your dataset into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual filename
df = pd.read_csv('your_dataset.csv')

# Assuming the target variable is named 'diagnosis'
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM model
svm_model = SVC(C=.1, degree=2, kernel='linear', random_state=42)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(svm_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('SVM Learning Curve')
plt.legend()
plt.show()

# Validation Curve
kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']  # Adjust the range based on your needs
train_scores, test_scores = validation_curve(svm_model, X_train, y_train, param_name='kernel', param_range=kernel_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(kernel_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(kernel_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xlabel('Kernel')
plt.ylabel('Accuracy Score')
plt.title('SVM Validation Curve')
plt.legend()
plt.show()

