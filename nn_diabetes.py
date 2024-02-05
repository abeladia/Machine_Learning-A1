import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load your dataset into a pandas DataFrame
train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['Diabetes_binary'], axis = 1), df['Diabetes_binary']

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)

# Define the Neural Network model
mlp_model = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(10,3), learning_rate='constant', random_state=42)

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
nn_model = MLPClassifier(random_state=42)
# Validation Curve for width
train_scores_width, test_scores_width = validation_curve(
    nn_model, X_train, y_train, param_name='hidden_layer_sizes', param_range=width_range, cv=5, scoring='accuracy'
)

# Validation Curve for depth (number of hidden layers)
depth_range = [1, 2, 3]
# Validation Curve for depth
train_scores_depth, test_scores_depth = validation_curve(
    nn_model, X_train, y_train, param_name='hidden_layer_sizes', param_range=depth_range, cv=5, scoring='accuracy'
)

# Plotting
plt.figure(figsize=(16, 6))

# Plot Validation Curve for Width
plt.subplot(1, 2, 1)
plt.plot(width_range, np.mean(train_scores_width, axis=1), label='Training Score')
plt.plot(width_range, np.mean(test_scores_width, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Nodes in Hidden Layer')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Validation Curve - Width Tuning')
plt.legend()

# Plot Validation Curve for Depth
plt.subplot(1, 2, 2)
plt.plot(depth_range, np.mean(train_scores_depth, axis=1), label='Training Score')
plt.plot(depth_range, np.mean(test_scores_depth, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Validation Curve - Depth Tuning')
plt.legend()
plt.tight_layout()
plt.show()