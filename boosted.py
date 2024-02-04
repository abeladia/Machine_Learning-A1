import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base decision tree model for AdaBoost
base_dt_model = DecisionTreeClassifier(max_depth=3)

# Define a range of hyperparameters to test
n_estimators_range = [50, 100, 200]
learning_rate_range = [0.01, 0.1, 1]

# Initialize lists to store results
train_accuracy_results = []
test_accuracy_results = []

# Manual hyperparameter tuning
for n_estimators in n_estimators_range:
    # Create AdaBoost model with specified hyperparameters
    adaboost_model = AdaBoostClassifier(base_dt_model, n_estimators=n_estimators, learning_rate=1.0, random_state=42)

    # Fit the model
    adaboost_model.fit(X_train, y_train)

    # Make predictions on training and test sets
    y_train_pred = adaboost_model.predict(X_train)
    y_test_pred = adaboost_model.predict(X_test)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_accuracy_results.append(train_accuracy)
    test_accuracy_results.append(test_accuracy)

# Convert results to DataFrame for visualization
results_df = pd.DataFrame({'Number of Estimators': n_estimators_range, 'Train Accuracy': train_accuracy_results, 'Test Accuracy': test_accuracy_results})

# Visualize results using line plots
plt.figure(figsize=(12, 6))

# Plot Number of Estimators vs Accuracy
plt.subplot(1, 2, 1)
plt.plot(results_df['Number of Estimators'], results_df['Train Accuracy'], label='Training Set', marker='o')
plt.plot(results_df['Number of Estimators'], results_df['Test Accuracy'], label='Test Set', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Number of Estimators vs Accuracy')
plt.legend()

# Plot Learning Rate vs Accuracy
plt.subplot(1, 2, 2)
for learning_rate in learning_rate_range:
    # Create AdaBoost model with specified learning rate
    adaboost_model = AdaBoostClassifier(base_dt_model, n_estimators=100, learning_rate=learning_rate, random_state=42)

    # Fit the model
    adaboost_model.fit(X_train, y_train)

    # Make predictions on test set
    y_test_pred = adaboost_model.predict(X_test)

    # Calculate accuracy scores
    test_accuracy = accuracy_score(y_test, y_test_pred)

    plt.plot(learning_rate, test_accuracy, marker='o', label=f'Learning Rate={learning_rate}')

plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Learning Rate vs Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
