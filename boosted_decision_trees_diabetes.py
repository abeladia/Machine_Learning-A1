import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset into a pandas DataFrame

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['Diabetes_binary'], axis = 1), df['Diabetes_binary']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base decision tree model for AdaBoost
base_dt_model = DecisionTreeClassifier(max_depth=3)
base_dt_model = base_dt_model.fit(X_train, y_train)
y_predict_pruned = base_dt_model.predict(X_test)

# Define the AdaBoost model
adaboost_model = AdaBoostClassifier(base_dt_model, random_state=42, n_estimators=25)
adaboost_model = adaboost_model.fit(X_train, y_train)
y_predict_boost = adaboost_model.predict(X_test)

target_names = ['Normal', 'Diabetes']
print("Decision tree:")
print(classification_report(y_test, y_predict_pruned, target_names=target_names))
print("Boost Decision tree:")
print(classification_report(y_test, y_predict_boost, target_names=target_names))

# Plot Learning Curve
plt.figure(figsize=(12, 6))
train_sizes, train_scores, test_scores = learning_curve(
    adaboost_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)
test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)

# Plotting Learning Curve
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_accuracy_mean, label='Training Accuracy', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")

plt.plot(train_sizes, test_accuracy_mean, label='Cross-validation Accuracy', marker='o')
plt.fill_between(train_sizes, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="g")

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('AdaBoost Learning Curve: Accuracy vs Training Size')
plt.legend()

# Plot Validation Curve
param_range = [10, 25, 50,75, 100, 150, 200]  # Example: Varying the number of weak learners (trees) in AdaBoost
param_name = "n_estimators"

train_scores, test_scores = validation_curve(
    adaboost_model, X_train, y_train,
    param_name=param_name, param_range=param_range, cv=5, scoring="accuracy", n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)
test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)

# Plotting Validation Curve
plt.subplot(1, 2, 2)
plt.plot(param_range, train_accuracy_mean, label="Training Accuracy", color="darkorange", marker='o')
plt.fill_between(param_range, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")

plt.plot(param_range, test_accuracy_mean, label="Cross-validation Accuracy", color="navy", marker='o')
plt.fill_between(param_range, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="g")

plt.xlabel(param_name)
plt.ylabel("Accuracy")
plt.title("AdaBoost Validation Curve: Accuracy vs Number of Weak Learners")
plt.legend()

plt.tight_layout()
plt.show()
