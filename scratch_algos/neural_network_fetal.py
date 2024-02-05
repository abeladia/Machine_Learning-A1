import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/fetal_health.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['fetal_health'], axis = 1), df['fetal_health']

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



# Define the Decision Tree model
nn_model = MLPClassifier(random_state=42)
nn_model = nn_model.fit(X_train, y_train)
y_predict = nn_model.predict(X_test)

nn_model_pruned = MLPClassifier()
nn_model_pruned = nn_model_pruned.fit(X_train, y_train)
y_predict_pruned = nn_model_pruned.predict(X_test)

target_names = ['Normal', 'Suspect', 'Pathological']

print("Decision tree:")
print(classification_report(y_test, y_predict, target_names=target_names))
print("Pruned:")
print(classification_report(y_test, y_predict_pruned, target_names=target_names))


# Plot Learning Curve
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    nn_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
train_sizes2, train_scores2, test_scores = learning_curve(
    nn_model_pruned, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)

train_accuracy_mean2 = np.mean(train_scores2, axis=1)
train_accuracy_std2 = np.std(train_scores2, axis=1)

test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)

# Plotting
plt.plot(train_sizes, train_accuracy_mean, label='Training Accuracy', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_accuracy_mean2, label='Training Accuracy with max-depth optimized', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean2 - train_accuracy_std2, train_accuracy_mean2 + train_accuracy_std2, alpha=0.1, color="b")
plt.plot(train_sizes, test_accuracy_mean, label='Testing Accuracy', marker='o')
plt.fill_between(train_sizes, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="g")

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Accuracy vs Training Size')
plt.legend()
plt.show()
