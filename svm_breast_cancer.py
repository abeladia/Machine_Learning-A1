import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the SVM model
svm_model = SVC(C=.1, degree=2, kernel='sigmoid', random_state=42)
random_forest_model = svm_model.fit(X_train, y_train)
y_predict = svm_model.predict(X_test)
print("Report:")
print(classification_report(y_test, y_predict))

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

# Validation Curve for 'linear' kernel
svm_model1 = SVC(C=.1, kernel='linear', degree=2, random_state=42)
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

# Validation Curve for 'sigmoid' kernel
param_range = np.logspace(-3, 3, 6)
svm_model2 = SVC(C=.1, kernel='sigmoid', degree=2, random_state=42)
train_scores, test_scores = validation_curve(svm_model2, X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy Score')
plt.title('SVM Validation Curve - Sigmoid Kernel')
plt.legend()
plt.show()