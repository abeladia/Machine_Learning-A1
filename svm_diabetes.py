import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['Diabetes_binary'], axis = 1), df['Diabetes_binary']

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM model
# svm_model = SVC(C=1, degree=2, kernel='rbf', random_state=42)
# # Learning Curve
# train_sizes, train_scores, test_scores = learning_curve(svm_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
#
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy Score')
# plt.title('SVM Learning Curve')
# plt.legend()
# plt.show()

# Validation Curve for 'linear' kernel
svm_model1 = SVC(C=1, degree=2, random_state=42)
param_range = np.logspace(-3, 3, 6)
train_scores, test_scores = validation_curve(svm_model1, X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy Score')
plt.title('SVM Validation Curve - Linear Kernel')
plt.legend()
plt.show()

# Validation Curve for 'rbf' kernel
param_range = np.logspace(-3, 3, 6)
train_scores, test_scores = validation_curve(svm_model1, X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy Score')
plt.title('SVM Validation Curve - RBF Kernel')
plt.legend()
plt.show()