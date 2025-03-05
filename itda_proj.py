# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convert data to numeric type
X = X.astype('float32')
y = y.astype('int32')

# Normalize the data
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for the neural network
X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=500)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)

# Print accuracies
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Confusion matrices
lr_cm = confusion_matrix(y_test, lr_predictions)
nn_cm = confusion_matrix(y_test, nn_predictions)

# Visualization
plt.figure(figsize=(20, 8))

# Logistic Regression Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Neural Network Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.pdf')
plt.close()

# Sample images with predictions
n_samples = 5
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

plt.figure(figsize=(20, 4))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, n_samples, i+1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    lr_pred = lr_model.predict([X_test[idx]])[0]
    nn_pred = nn_model.predict([X_test[idx]])[0]
    plt.title(f"LR: {lr_pred}, NN: {nn_pred}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.pdf')
plt.close()

# Learning curves
train_sizes = np.linspace(0.1, 1.0, 5)
train_sizes, train_scores_lr, test_scores_lr = learning_curve(
    LogisticRegression(random_state=42, max_iter=500), X_train, y_train,
    train_sizes=train_sizes, cv=5, n_jobs=-1)

train_sizes, train_scores_nn, test_scores_nn = learning_curve(
    MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=-1)

plt.figure(figsize=(20, 8))

# Logistic Regression Learning Curve
plt.subplot(1, 2, 1)
plt.plot(train_sizes, np.mean(train_scores_lr, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores_lr, axis=1), 'o-', label='Cross-validation score')
plt.title('Logistic Regression Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')

# Neural Network Learning Curve
plt.subplot(1, 2, 2)
plt.plot(train_sizes, np.mean(train_scores_nn, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores_nn, axis=1), 'o-', label='Cross-validation score')
plt.title('Neural Network Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('learning_curves.pdf')
plt.close()

# Generate classification reports
lr_report = classification_report(y_test, lr_predictions)
nn_report = classification_report(y_test, nn_predictions)

# Save classification reports to a file
with open('classification_reports.txt', 'w') as f:
    f.write("Logistic Regression Classification Report:\n")
    f.write(lr_report)
    f.write("\n\nNeural Network Classification Report:\n")
    f.write(nn_report)

print("Analysis complete. Results saved as PDF files and text file.")
