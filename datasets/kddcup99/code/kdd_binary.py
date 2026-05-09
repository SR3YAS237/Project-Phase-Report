import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# LOAD DATASET
# =====================================
print("Loading dataset...")

file_path = "kddcup.data.gz"

chunks = pd.read_csv(
    file_path,
    header=None,
    chunksize=100000,
    compression='gzip'
)

df_list = []

for i, chunk in enumerate(chunks):

    df_list.append(chunk)

    # LOAD ~300K ROWS
    if i == 2:
        break

df = pd.concat(df_list, ignore_index=True)

print("Loaded shape:", df.shape)

# =====================================
# COLUMN NAMES
# =====================================
columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
"hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
"count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label"
]

df.columns = columns

# =====================================
# BINARY LABEL MAPPING
# =====================================
df['label'] = df['label'].apply(
    lambda x: 'normal' if x == 'normal.' else 'attack'
)

# =====================================
# ENCODE CATEGORICAL FEATURES
# =====================================
df = pd.get_dummies(
    df,
    columns=['protocol_type', 'service', 'flag']
)

# =====================================
# LABEL ENCODING
# =====================================
le = LabelEncoder()

y = le.fit_transform(df['label'])

# =====================================
# FEATURES
# =====================================
X = df.drop('label', axis=1)

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# NORMALIZATION
# =====================================
scaler = Normalizer()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# RANDOM FOREST
# =====================================
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRF Accuracy:", rf_accuracy)

rf_report = classification_report(
    y_test,
    y_pred_rf,
    target_names=le.classes_
)

rf_cm = confusion_matrix(
    y_test,
    y_pred_rf
)

# =====================================
# SVM
# =====================================
print("\nTraining SVM...")

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:", svm_accuracy)

svm_report = classification_report(
    y_test,
    y_pred_svm,
    target_names=le.classes_
)

svm_cm = confusion_matrix(
    y_test,
    y_pred_svm
)

# =====================================
# DNN
# =====================================
print("\nTraining DNN...")

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
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_pred_nn = (
    model_nn.predict(X_test) > 0.5
).astype(int)

dnn_accuracy = accuracy_score(y_test, y_pred_nn)

print("\nDNN Accuracy:", dnn_accuracy)

dnn_report = classification_report(
    y_test,
    y_pred_nn,
    target_names=le.classes_
)

dnn_cm = confusion_matrix(
    y_test,
    y_pred_nn
)

# =====================================
# SAVE COMBINED RESULTS
# =====================================
with open("kdd_results_binary.txt", "w") as f:

    f.write("========== KDD BINARY RESULTS ==========\n\n")

    # RANDOM FOREST
    f.write("===== RANDOM FOREST =====\n\n")

    f.write(f"Accuracy: {rf_accuracy}\n\n")

    f.write(rf_report)

    f.write("\nConfusion Matrix:\n")

    f.write(str(rf_cm))

    # SVM
    f.write("\n\n===== SVM =====\n\n")

    f.write(f"Accuracy: {svm_accuracy}\n\n")

    f.write(svm_report)

    f.write("\nConfusion Matrix:\n")

    f.write(str(svm_cm))

    # DNN
    f.write("\n\n===== DNN =====\n\n")

    f.write(f"Accuracy: {dnn_accuracy}\n\n")

    f.write(dnn_report)

    f.write("\nConfusion Matrix:\n")

    f.write(str(dnn_cm))

print("✅ Combined results saved")

# =====================================
# RF CONFUSION MATRIX GRAPH
# =====================================
plt.figure(figsize=(6,5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("RF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("kdd_rf_confusion_binary.png")

plt.close()

# =====================================
# SVM CONFUSION MATRIX GRAPH
# =====================================
plt.figure(figsize=(6,5))

sns.heatmap(
    svm_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("kdd_svm_confusion_binary.png")

plt.close()

# =====================================
# DNN CONFUSION MATRIX GRAPH
# =====================================
plt.figure(figsize=(6,5))

sns.heatmap(
    dnn_cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("DNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("kdd_dnn_confusion_binary.png")

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

plt.figure(figsize=(5,5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("KDD Binary Accuracy Comparison")

plt.savefig("kdd_accuracy_comparison_binary.png")

plt.close()

print("✅ Graphs saved")

print("\n🎯 KDD BINARY EXPERIMENT COMPLETED!")