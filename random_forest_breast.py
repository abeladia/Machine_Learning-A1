import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest model
random_forest_model = RandomForestClassifier(max_depth=2, n_estimators=1500, random_state=42)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(random_forest_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Random Forest Learning Curve')
plt.legend()
plt.show()

# Validation Curve
param_range = [100, 200, 500, 1000, 1500, 2000, 5000]  # Adjust the range based on your needs
train_scores, test_scores = validation_curve(random_forest_model, X_train, y_train, param_name='n_estimators', param_range=param_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy Score')
plt.title('Random Forest Validation Curve')
plt.legend()
plt.show()
