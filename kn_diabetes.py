import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load your dataset into a pandas DataFrame
train_data = '/Users/anishabeladia/IdeaProjects/ML-A1/sample_data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'

df = pd.read_csv(train_data)

X,y = df.drop(['Diabetes_binary'], axis = 1), df['Diabetes_binary']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Define the KNN model
knn_model = KNeighborsClassifier(n_neighbors=9, weights='uniform')

# Fit the model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_predict_knn = knn_model.predict(X_test)

# Display classification report
print("Report:")
print(classification_report(y_test, y_predict_knn))

# Learning Curve
train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(knn_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_knn, np.mean(train_scores_knn, axis=1), label='Training Score')
plt.plot(train_sizes_knn, np.mean(test_scores_knn, axis=1), label='Cross-validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('KNN Learning Curve')
plt.legend()
plt.show()

# Validation Curve
knn_model = KNeighborsClassifier()
k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17]  # Adjust the range based on your needs
train_scores_knn, test_scores_knn = validation_curve(knn_model, X_train, y_train, param_name='n_neighbors', param_range=k_range, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(k_range, np.mean(train_scores_knn, axis=1), label='Training Score')
plt.plot(k_range, np.mean(test_scores_knn, axis=1), label='Cross-validation Score')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.title('KNN Validation Curve')
plt.legend()
plt.show()
