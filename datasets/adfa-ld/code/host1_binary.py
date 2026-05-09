import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# DATASET PATHS
# =====================================
NORMAL_PATH = "ADFA-LD/Training_Data_Master"

ATTACK_PATH = "ADFA-LD/Attack_Data_Master"

# =====================================
# LOAD NORMAL DATA
# =====================================
print("Loading normal traces...")

normal_data = []

for root, dirs, files in os.walk(NORMAL_PATH):

    for file in files:

        filepath = os.path.join(root, file)

        try:
            with open(filepath, 'r') as f:

                content = f.read().strip()

                normal_data.append(content)

        except:
            pass

print("Normal samples:", len(normal_data))

# =====================================
# LOAD ATTACK DATA
# =====================================
print("\nLoading attack traces...")

attack_data = []

for root, dirs, files in os.walk(ATTACK_PATH):

    for file in files:

        filepath = os.path.join(root, file)

        try:
            with open(filepath, 'r') as f:

                content = f.read().strip()

                attack_data.append(content)

        except:
            pass

print("Attack samples:", len(attack_data))

# =====================================
# CREATE DATAFRAME
# =====================================
texts = normal_data + attack_data

labels = [0] * len(normal_data) + [1] * len(attack_data)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print("\nDataset Shape:", df.shape)

# =====================================
# TF-IDF FEATURE EXTRACTION
# =====================================
print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=3000
)

X = vectorizer.fit_transform(df['text']).toarray()

y = df['label']

print("Feature Shape:", X.shape)

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================
# RANDOM FOREST
# =====================================
print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

rf_precision = precision_score(y_test, y_pred_rf)

rf_recall = recall_score(y_test, y_pred_rf)

rf_f1 = f1_score(y_test, y_pred_rf)

print("\nRF Accuracy:", rf_accuracy)

# =====================================
# SAVE RF RESULTS
# =====================================
with open("rf_adfa_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== RANDOM FOREST =====\n\n")

    f.write(f"Accuracy : {rf_accuracy}\n")
    f.write(f"Precision: {rf_precision}\n")
    f.write(f"Recall   : {rf_recall}\n")
    f.write(f"F1 Score : {rf_f1}\n\n")

    f.write("Classification Report:\n")

    f.write(classification_report(y_test, y_pred_rf))

    f.write("\nConfusion Matrix:\n")

    f.write(str(confusion_matrix(y_test, y_pred_rf)))

print("✅ RF results saved")

# =====================================
# RF CONFUSION MATRIX
# =====================================
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title("RF Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("rf_adfa_binary_confusion.png")

plt.close()

# =====================================
# SVM
# =====================================
print("\n==============================")
print("TRAINING SVM")
print("==============================")

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

svm_precision = precision_score(y_test, y_pred_svm)

svm_recall = recall_score(y_test, y_pred_svm)

svm_f1 = f1_score(y_test, y_pred_svm)

print("\nSVM Accuracy:", svm_accuracy)

# =====================================
# SAVE SVM RESULTS
# =====================================
with open("svm_adfa_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== SVM =====\n\n")

    f.write(f"Accuracy : {svm_accuracy}\n")
    f.write(f"Precision: {svm_precision}\n")
    f.write(f"Recall   : {svm_recall}\n")
    f.write(f"F1 Score : {svm_f1}\n\n")

    f.write("Classification Report:\n")

    f.write(classification_report(y_test, y_pred_svm))

    f.write("\nConfusion Matrix:\n")

    f.write(str(confusion_matrix(y_test, y_pred_svm)))

print("✅ SVM results saved")

# =====================================
# SVM CONFUSION MATRIX
# =====================================
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("svm_adfa_binary_confusion.png")

plt.close()

# =====================================
# DNN
# =====================================
print("\n==============================")
print("TRAINING DNN")
print("==============================")

model_nn = Sequential([

    Input(shape=(X_train.shape[1],)),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model_nn.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_prob_nn = model_nn.predict(X_test)

y_pred_nn = (y_prob_nn > 0.5).astype(int)

dnn_accuracy = accuracy_score(y_test, y_pred_nn)

dnn_precision = precision_score(y_test, y_pred_nn)

dnn_recall = recall_score(y_test, y_pred_nn)

dnn_f1 = f1_score(y_test, y_pred_nn)

print("\nDNN Accuracy:", dnn_accuracy)

# =====================================
# SAVE DNN RESULTS
# =====================================
with open("dnn_adfa_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== DNN =====\n\n")

    f.write(f"Accuracy : {dnn_accuracy}\n")
    f.write(f"Precision: {dnn_precision}\n")
    f.write(f"Recall   : {dnn_recall}\n")
    f.write(f"F1 Score : {dnn_f1}\n\n")

    f.write("Classification Report:\n")

    f.write(classification_report(y_test, y_pred_nn))

    f.write("\nConfusion Matrix:\n")

    f.write(str(confusion_matrix(y_test, y_pred_nn)))

print("✅ DNN results saved")

# =====================================
# DNN CONFUSION MATRIX
# =====================================
cm_dnn = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_dnn,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title("DNN Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("dnn_adfa_binary_confusion.png")

plt.close()

# =====================================
# ACCURACY COMPARISON GRAPH
# =====================================
models = ['RF', 'SVM', 'DNN']

accuracies = [
    rf_accuracy,
    svm_accuracy,
    dnn_accuracy
]

plt.figure(figsize=(6,5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")

plt.title("ADFA-LD Binary Classification Accuracy")

plt.savefig("adfa_binary_accuracy_comparison.png")

plt.close()

print("✅ Accuracy comparison graph saved")

print("\n🎯 ADFA-LD BINARY CLASSIFICATION COMPLETED!")