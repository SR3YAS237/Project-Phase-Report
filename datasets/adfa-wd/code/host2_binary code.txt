import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
# HOST LOGS PATH
# =====================================
HOST_LOGS_PATH = "ADFA-IDS_DATASETS/NGIDS-DS/NGIDS-DS/host logs"

# =====================================
# LOAD FILES
# =====================================
print("Loading host log files...")

df_list = []

for file in os.listdir(HOST_LOGS_PATH):

    filepath = os.path.join(HOST_LOGS_PATH, file)

    try:

        # CSV FILES
        if file.endswith(".csv"):

            temp_df = pd.read_csv(
                filepath,
                header=None
            )

            df_list.append(temp_df)

        # EXCEL FILES
        elif file.endswith(".xlsx") or file.endswith(".xls"):

            temp_df = pd.read_excel(
                filepath,
                header=None
            )

            df_list.append(temp_df)

    except Exception as e:

        print(f"Error reading {file}: {e}")

# =====================================
# COMBINE DATA
# =====================================
df = pd.concat(df_list, ignore_index=True)

print("\nOriginal Shape:", df.shape)

# =====================================
# SAMPLE DATASET
# =====================================
print("\nSampling dataset...")

df = df.sample(
    n=500000,
    random_state=42
)

print("Sampled Shape:", df.shape)

# =====================================
# CLEAN DATA
# =====================================
print("\nCleaning dataset...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)

# =====================================
# LABEL COLUMN
# =====================================
label_col = 6

# =====================================
# BINARY LABEL MAPPING
# =====================================
print("\nConverting labels...")

df[label_col] = df[label_col].apply(
    lambda x: 0 if str(x).lower() == "normal" else 1
)

print(df[label_col].value_counts())

# =====================================
# FEATURES & LABELS
# =====================================
X = df.drop(columns=[label_col])

y = df[label_col]

# =====================================
# ENCODE CATEGORICAL FEATURES
# =====================================
print("\nEncoding categorical columns...")

for col in X.columns:

    if X[col].dtype == 'object':

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

# =====================================
# NORMALIZATION
# =====================================
print("\nScaling features...")

scaler = StandardScaler()

X = scaler.fit_transform(X)

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
    n_estimators=50,
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
with open("rf_adfawd_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== RANDOM FOREST =====\n\n")

    f.write(f"Accuracy : {rf_accuracy}\n")
    f.write(f"Precision: {rf_precision}\n")
    f.write(f"Recall   : {rf_recall}\n")
    f.write(f"F1 Score : {rf_f1}\n\n")

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
    cmap='Blues'
)

plt.title("RF Confusion Matrix")

plt.savefig("rf_adfawd_binary_confusion.png")

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
with open("svm_adfawd_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== SVM =====\n\n")

    f.write(f"Accuracy : {svm_accuracy}\n")
    f.write(f"Precision: {svm_precision}\n")
    f.write(f"Recall   : {svm_recall}\n")
    f.write(f"F1 Score : {svm_f1}\n\n")

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
    cmap='Greens'
)

plt.title("SVM Confusion Matrix")

plt.savefig("svm_adfawd_binary_confusion.png")

plt.close()

# =====================================
# DNN
# =====================================
print("\n==============================")
print("TRAINING DNN")
print("==============================")

model_nn = Sequential([

    Input(shape=(X_train.shape[1],)),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
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
    batch_size=256,
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
with open("dnn_adfawd_binary.txt", "w", encoding="utf-8") as f:

    f.write("===== DNN =====\n\n")

    f.write(f"Accuracy : {dnn_accuracy}\n")
    f.write(f"Precision: {dnn_precision}\n")
    f.write(f"Recall   : {dnn_recall}\n")
    f.write(f"F1 Score : {dnn_f1}\n\n")

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
    cmap='Reds'
)

plt.title("DNN Confusion Matrix")

plt.savefig("dnn_adfawd_binary_confusion.png")

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

plt.title("ADFA-WD Binary Accuracy")

plt.savefig("adfawd_binary_accuracy_comparison.png")

plt.close()

print("✅ Accuracy comparison graph saved")

print("\n🎯 ADFA-WD BINARY CLASSIFICATION COMPLETED!")