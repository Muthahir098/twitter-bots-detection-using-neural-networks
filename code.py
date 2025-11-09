# Importing all the necessary libraries
import sys
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Fix for UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the dataset into a DataFrame
data = pd.read_csv('D:/Python/reduced_bot_detection_data.csv')

# Separating the features (X) and the target labels (y)
X = data[['Retweet Count', 'Mention Count', 'Follower Count']]
y = data['Bot Label']
usernames = data['Username']  # we'll use this for tracking predictions later

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test, usernames_train, usernames_test = train_test_split(
    X, y, usernames, test_size=0.5, random_state=42
)

# Standardizing the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balancing the training data using resampling
X_train_resampled, y_train_resampled = resample(
    X_train_scaled, y_train, replace=True, n_samples=len(X_train), random_state=42
)

# Building the neural network architecture
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Adding callbacks to monitor training and prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Training the model
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=100, batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Making predictions on the test set
predictions = model.predict(X_test_scaled)
# Convert predictions to binary values (0 or 1)
predictions = (predictions > 0.5).astype(int)

# Printing model performance metrics
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Plotting the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualizing the confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not a Bot', 'Bot'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Plotting the ROC curve and calculating AUC
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal reference line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
