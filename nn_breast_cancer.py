import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load your dataset into a pandas DataFrame

train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
train_data["diagnosis"] = train_data["diagnosis"].replace({'M':1,'B':0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Neural Network model
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(mlp_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Learning Curve')
plt.legend()
plt.show()
# Validation Curve for width (number of nodes within a layer)
width_range = [10, 50, 100]
depth_fixed = 2  # Fixing depth at 2

param_grid_width = [(width, depth_fixed) for width in width_range]

nn_model = MLPClassifier(random_state=42)
# Validation Curve for width
train_scores_width, test_scores_width = validation_curve(
    nn_model, X_train, y_train, param_name='hidden_layer_sizes', param_range=param_grid_width, cv=5, scoring='accuracy'
)

# Convert the parameter values to strings for plotting
param_range_str_width = [f'({width},{depth_fixed})' for width, _ in param_grid_width]

# Validation Curve for depth (number of hidden layers)
depth_range = [1, 2, 3]
width_fixed = 50  # Fixing width at 50

param_grid_depth = [(width_fixed, depth) for depth in depth_range]

# Validation Curve for depth
train_scores_depth, test_scores_depth = validation_curve(
    nn_model, X_train, y_train, param_name='hidden_layer_sizes', param_range=param_grid_depth, cv=5, scoring='accuracy'
)

# Convert the parameter values to strings for plotting
param_range_str_depth = [f'({width_fixed},{depth})' for _, depth in param_grid_depth]

# Plotting
plt.figure(figsize=(16, 6))

# Plot Validation Curve for Width
plt.subplot(1, 2, 1)
plt.plot(param_range_str_width, np.mean(train_scores_width, axis=1), label='Training Score')
plt.plot(param_range_str_width, np.mean(test_scores_width, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Nodes in Hidden Layer')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Validation Curve - Width Tuning')
plt.legend()

# Plot Validation Curve for Depth
plt.subplot(1, 2, 2)
plt.plot(param_range_str_depth, np.mean(train_scores_depth, axis=1), label='Training Score')
plt.plot(param_range_str_depth, np.mean(test_scores_depth, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Validation Curve - Depth Tuning')
plt.legend()

plt.tight_layout()
plt.show()