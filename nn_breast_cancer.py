import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time

# Load your dataset into a pandas DataFrame
train_data = './sample_data/breast-cancer-wisconsin-data.csv'

df = pd.read_csv(train_data)
train_data= df.drop(columns=['id'],axis=1)
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])
# train_data["diagnosis"] = train_data["diagnosis"].map({'M': 1, 'B': 0})

X,y = df.drop(['diagnosis'], axis = 1), df['diagnosis']

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Define the Neural Network model with early stopping
model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=3, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Neural Network Training and Validation Loss')
plt.legend()
plt.show()

# Define the Neural Network model
mlp_model = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(50,), learning_rate='constant', random_state=42)

start_time = time.time()
mlp_model = mlp_model.fit(X_train, y_train)
training_time = time.time() - start_time
print(training_time)

y_predict = mlp_model.predict(X_test)
print("Report:")
print(classification_report(y_test, y_predict))

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


# Learning Curve with max_iter variation
max_iter_range = [50, 100, 200, 400]
plt.figure(figsize=(10, 6))

for max_iter in max_iter_range:
    mlp_model.set_params(max_iter=max_iter)
    train_sizes, train_scores, test_scores = learning_curve(mlp_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
    plt.plot(np.mean(train_scores, axis=1), label=f'Training Curve (max_iter={max_iter})')

plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Neural Network Learning Curve - Iterations')
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