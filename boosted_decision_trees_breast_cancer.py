import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss  # Import zero_one_loss for error rate
from sklearn.metrics import confusion_matrix, classification_report

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base decision tree model for AdaBoost
base_dt_model = DecisionTreeClassifier(max_depth=2)

# Define a range of hyperparameters to test
n_estimators_range = [25, 50, 100, 200]
learning_rate_range = [0.01, 0.1, 1, ]

# Initialize lists to store results
train_error_results = []
test_error_results = []

# Manual hyperparameter tuning
for n_estimators in n_estimators_range:
    # Create AdaBoost model with specified hyperparameters
    adaboost_model = AdaBoostClassifier(base_dt_model, n_estimators=n_estimators, random_state=42)

    # Fit the model
    adaboost_model.fit(X_train, y_train)

    # Make predictions on training and test sets
    y_train_pred = adaboost_model.predict(X_train)
    y_test_pred = adaboost_model.predict(X_test)

    # Calculate error rates
    train_error = zero_one_loss(y_train, y_train_pred)
    test_error = zero_one_loss(y_test, y_test_pred)

    train_error_results.append(train_error)
    test_error_results.append(test_error)

# Convert results to DataFrame for visualization
results_df = pd.DataFrame({'Number of Estimators': n_estimators_range, 'Train Error Rate': train_error_results, 'Test Error Rate': test_error_results})

# Visualize results using line plots
plt.figure(figsize=(12, 6))

# Plot Number of Estimators vs Error Rate
plt.subplot(1, 2, 1)
plt.plot(results_df['Number of Estimators'], results_df['Train Error Rate'], label='Training Set', marker='o')
plt.plot(results_df['Number of Estimators'], results_df['Test Error Rate'], label='Test Set', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('Number of Estimators vs Error Rate')
plt.legend()

# Initialize lists to store results
train_error_results2 = []
test_error_results2 = []

# Plot Learning Rate vs Error Rate
plt.subplot(1, 2, 2)
for learning_rate in learning_rate_range:
    # Create AdaBoost model with specified learning rate
    adaboost_model = AdaBoostClassifier(base_dt_model, learning_rate=learning_rate, random_state=42)
    # Fit the model
    adaboost_model.fit(X_train, y_train)

    # Make predictions on training and test sets
    y_train_pred2 = adaboost_model.predict(X_train)
    y_test_pred2 = adaboost_model.predict(X_test)

    # Calculate error rates
    train_error2 = zero_one_loss(y_train, y_train_pred2)
    test_error2 = zero_one_loss(y_test, y_test_pred2)

    train_error_results2.append(train_error2)
    test_error_results2.append(test_error2)

# Convert results to DataFrame for visualization
results_df2 = pd.DataFrame({'Learning Rate Range': learning_rate_range, 'Train Error Rate': train_error_results2, 'Test Error Rate': test_error_results2})

plt.plot(results_df2['Learning Rate Range'], results_df2['Train Error Rate'], label='Training Set', marker='o')
plt.plot(results_df2['Learning Rate Range'], results_df2['Test Error Rate'], label='Test Set', marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Error Rate')
plt.title('Learning Rate vs Error Rate')
plt.legend()

plt.tight_layout()
plt.show()

# Define the base decision tree model for AdaBoost
base_dt = AdaBoostClassifier(base_dt_model)
base_dt = base_dt.fit(X_train, y_train)
y_predict_pruned = base_dt.predict(X_test)

# Define the AdaBoost model
adaboost_model_tuned = AdaBoostClassifier(base_dt_model, n_estimators=200, learning_rate=.01)
adaboost_model_tuned = adaboost_model_tuned.fit(X_train, y_train)
y_predict_boost = adaboost_model_tuned.predict(X_test)

target_names = ['Malignant', 'Benign']
print("Decision tree:")
print(classification_report(y_test, y_predict_pruned, target_names=target_names))
print("Boost Decision tree:")
print(classification_report(y_test, y_predict_boost, target_names=target_names))

# Plot Learning Curve
plt.figure(figsize=(12, 6))
train_sizes, train_scores, test_scores = learning_curve(
    base_dt_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
train_sizes2, train_scores2, test_scores2 = learning_curve(
    adaboost_model_tuned, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)

train_accuracy_mean2 = np.mean(train_scores2, axis=1)
train_accuracy_std2 = np.std(train_scores2, axis=1)

test_accuracy_mean1 = np.mean(test_scores, axis=1)
test_accuracy_std1 = np.std(test_scores, axis=1)

test_accuracy_mean = np.mean(test_scores2, axis=1)
test_accuracy_std = np.std(test_scores2, axis=1)

# Plotting Learning Curve
plt.subplot(1, 2, 1)
#plt.plot(train_sizes, train_accuracy_mean, label="Training Accuracy of AdaBoost", color="darkorange", marker='o')
#plt.fill_between(train_sizes, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")

plt.plot(train_sizes, train_accuracy_mean2, label='Training Accuracy with AdaBoost-tuned',color="magenta", marker='o')
#plt.fill_between(train_sizes, train_accuracy_mean2 - train_accuracy_std2, train_accuracy_mean2 + train_accuracy_std2, alpha=0.1, color="m")

plt.plot(train_sizes, test_accuracy_mean, label="Cross-validation Accuracy of AdaBoost-tuned", color="navy", marker='o')
#plt.fill_between(train_sizes, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="b")

#plt.plot(train_sizes, test_accuracy_mean, label="Cross-validation Accuracy of AdaBoost", color="red", marker='o')
#plt.fill_between(train_sizes, test_accuracy_mean1 - test_accuracy_std1, test_accuracy_mean1 + test_accuracy_std1, alpha=0.1, color="g")


plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('AdaBoost Learning Curve: Accuracy vs Training Size')
plt.legend()
plt.show()

