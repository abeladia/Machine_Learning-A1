import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load your dataset into a pandas DataFrame
train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['Diabetes_binary'], axis = 1), df['Diabetes_binary']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Neural Network model
nn_model = MLPClassifier(random_state=42)

# Plot Learning Curve
plt.figure(figsize=(18, 6))
train_sizes, train_scores, test_scores = learning_curve(
    nn_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)
test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)

# Plotting Learning Curve
plt.subplot(1, 3, 1)
plt.plot(train_sizes, train_accuracy_mean, label='Training Accuracy', marker='o')
plt.fill_between(train_sizes, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1, color="r")

plt.plot(train_sizes, test_accuracy_mean, label='Testing Accuracy', marker='o')
plt.fill_between(train_sizes, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1, color="g")

plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Neural Network Learning Curve: Accuracy vs Training Size')
plt.legend()

# Plot Validation Curve for Width Tuning
param_range_width = [10, 50, 100, 200]
train_scores_width, test_scores_width = validation_curve(
    nn_model, X_train, y_train,
    param_name="hidden_layer_sizes", param_range=param_range_width, cv=5, scoring="accuracy", n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean_width = np.mean(train_scores_width, axis=1)
train_accuracy_std_width = np.std(train_scores_width, axis=1)
test_accuracy_mean_width = np.mean(test_scores_width, axis=1)
test_accuracy_std_width = np.std(test_scores_width, axis=1)

# Plotting Validation Curve for Width Tuning
plt.subplot(1, 3, 2)
plt.plot(param_range_width, train_accuracy_mean_width, label="Training Accuracy", color="darkorange", marker='o')
plt.fill_between(param_range_width, train_accuracy_mean_width - train_accuracy_std_width, train_accuracy_mean_width + train_accuracy_std_width, alpha=0.1, color="r")

plt.plot(param_range_width, test_accuracy_mean_width, label="Testing Accuracy", color="navy", marker='o')
plt.fill_between(param_range_width, test_accuracy_mean_width - test_accuracy_std_width, test_accuracy_mean_width + test_accuracy_std_width, alpha=0.1, color="g")

plt.xlabel("Hidden Layer Sizes (Width)")
plt.ylabel("Accuracy")
plt.title("Neural Network Validation Curve: Accuracy vs Width")
plt.legend()

# Plot Validation Curve for Depth Tuning
param_range_depth = [1, 2, 3, 4]
train_scores_depth, test_scores_depth = validation_curve(
    nn_model, X_train, y_train,
    param_name="hidden_layer_sizes", param_range=[(50,) * d for d in param_range_depth], cv=5, scoring="accuracy", n_jobs=-1
)

# Calculate mean and standard deviation of accuracy scores
train_accuracy_mean_depth = np.mean(train_scores_depth, axis=1)
train_accuracy_std_depth = np.std(train_scores_depth, axis=1)
test_accuracy_mean_depth = np.mean(test_scores_depth, axis=1)
test_accuracy_std_depth = np.std(test_scores_depth, axis=1)

# Plotting Validation Curve for Depth Tuning
plt.subplot(1, 3, 3)
plt.plot(param_range_depth, train_accuracy_mean_depth, label="Training Accuracy", color="darkorange", marker='o')
plt.fill_between(param_range_depth, train_accuracy_mean_depth - train_accuracy_std_depth, train_accuracy_mean_depth + train_accuracy_std_depth, alpha=0.1, color="r")

plt.plot(param_range_depth, test_accuracy_mean_depth, label="Testing Accuracy", color="navy", marker='o')
plt.fill_between(param_range_depth, test_accuracy_mean_depth - test_accuracy_std_depth, test_accuracy_mean_depth + test_accuracy_std_depth, alpha=0.1, color="g")

plt.xlabel("Number of Hidden Layers (Depth)")
plt.ylabel("Accuracy")
plt.title("Neural Network Validation Curve: Accuracy vs Depth")
plt.legend()

plt.tight_layout()
plt.show()