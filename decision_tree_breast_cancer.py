import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from matplotlib import pyplot as plt


train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':2})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

train_errors = []
test_errors = []

# Vary max depth from 1 to 20
max_depth_values = range(1, 15)

# Train Decision Tree for each max depth value
for max_depth in max_depth_values:
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)

    # Predictions on training and testing sets
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)

    # Calculate error rates
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    # Append errors to arrays
    train_errors.append(train_error)
    test_errors.append(test_error)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, train_errors, label='Training Error', marker='o')
plt.plot(max_depth_values, test_errors, label='Testing Error', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Error Rate')
plt.title('Training and Testing Error Rates as a Function of Max Depth')
plt.legend()
plt.show()

train_errors1 = []
test_errors1 = []

# Vary max depth from 1 to 20
max_cost_values = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, 1]

# Train Decision Tree for each max depth value
for max_depth in max_cost_values:
    dt_model = DecisionTreeClassifier(ccp_alpha=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)

    # Predictions on training and testing sets
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)

    # Calculate error rates
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    # Append errors to arrays
    train_errors1.append(train_error)
    test_errors1.append(test_error)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(max_cost_values, train_errors1, label='Training Error', marker='o')
plt.plot(max_cost_values, test_errors1, label='Testing Error', marker='o')
plt.xlabel('Cost')
plt.ylabel('Error Rate')
plt.title('Training and Testing Error Rates as a Function of Cost')
plt.legend()
plt.show()

# Define the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model = dt_model.fit(X_train, y_train)
y_predict = dt_model.predict(X_test)

dt_model_pruned = DecisionTreeClassifier(max_depth=2)
dt_model_pruned = dt_model_pruned.fit(X_train, y_train)
y_predict_pruned = dt_model_pruned.predict(X_test)

dt_model_post_pruned = DecisionTreeClassifier(ccp_alpha=.002)
dt_model_post_pruned = dt_model_post_pruned.fit(X_train, y_train)
y_predict_post_pruned = dt_model_post_pruned.predict(X_test)

target_names = ['Malignant', 'Benign']

print("Decision tree:")
print(classification_report(y_test, y_predict, target_names=target_names))
print("Pruned:")
print(classification_report(y_test, y_predict_pruned, target_names=target_names))
print("Post-Pruned:")
print(classification_report(y_test, y_predict_post_pruned, target_names=target_names))


# Plot Learning Curve
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    dt_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
train_sizes2, train_scores2, test_scores = learning_curve(
    dt_model_pruned, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_sizes3, train_scores3, test_scores = learning_curve(
    dt_model_post_pruned, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)

train_accuracy_mean2 = np.mean(train_scores2, axis=1)
train_accuracy_std2 = np.std(train_scores2, axis=1)

train_accuracy_mean3 = np.mean(train_scores3, axis=1)
train_accuracy_std3 = np.std(train_scores3, axis=1)

test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)

# Plotting
plt.plot(train_sizes, train_accuracy_mean, label='Training Accuracy', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_accuracy_mean2, label='Training Accuracy with max-depth optimized', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean2 - train_accuracy_std2, train_accuracy_mean2 + train_accuracy_std2, alpha=0.1, color="b")
plt.plot(train_sizes, train_accuracy_mean2, label='Training Accuracy with cost-complexity optimized', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean2 - train_accuracy_std3, train_accuracy_mean3 + train_accuracy_std3, alpha=0.1, color="m")
plt.plot(train_sizes, test_accuracy_mean, label='Testing Accuracy', marker='o')
plt.fill_between(train_sizes, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="g")

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Accuracy vs Training Size')
plt.legend()
plt.show()
