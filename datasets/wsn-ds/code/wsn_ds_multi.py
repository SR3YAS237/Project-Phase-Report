import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
# LOAD DATASET
# =====================================
print("Loading WSN-DS dataset...")
df = pd.read_csv('../../WSN-DS.csv')
df.columns = df.columns.str.strip()

print("Original Shape:", df.shape)

# =====================================
# CLEAN DATA
# =====================================
print("\nCleaning dataset...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("Shape after cleaning:", df.shape)



# =====================================
# ENCODE LABELS
# =====================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Attack type'])

# =====================================
# FEATURES
# =====================================
X = df.drop('Attack type', axis=1)

# =====================================
# ENCODE CATEGORICAL FEATURES
# =====================================
print("\nEncoding categorical columns...")
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

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
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================
# RANDOM FOREST
# =====================================
print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")

rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRF Accuracy:", rf_accuracy)

# =====================================
# SVM
# =====================================
print("\n==============================")
print("TRAINING SVM")
print("==============================")

svm_model = LinearSVC(max_iter=1000, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", svm_accuracy)

# =====================================
# DNN
# =====================================
print("\n==============================")
print("TRAINING DNN")
print("==============================")

num_classes = len(np.unique(y_train))
model_nn = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model_nn.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

model_nn.fit(
    X_train, y_train, epochs=10, batch_size=256,
    validation_split=0.2, callbacks=[early_stop], verbose=1
)

y_pred_nn = np.argmax(model_nn.predict(X_test), axis=1)

dnn_accuracy = accuracy_score(y_test, y_pred_nn)
print("\nDNN Accuracy:", dnn_accuracy)

# =====================================
# SAVE COMBINED RESULTS
# =====================================
with open("../result/wsn_ds_results_multi.txt", "w") as f:
    f.write("========== WSN-DS MULTI RESULTS ==========\n\n")
    
    # RF
    f.write("===== RANDOM FOREST =====\n\n")
    f.write(f"Accuracy: {rf_accuracy}\n\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_.astype(str)))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_rf)))
    
    # SVM
    f.write("\n\n===== SVM =====\n\n")
    f.write(f"Accuracy: {svm_accuracy}\n\n")
    f.write(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_.astype(str)))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_svm)))
    
    # DNN
    f.write("\n\n===== DNN =====\n\n")
    f.write(f"Accuracy: {dnn_accuracy}\n\n")
    f.write(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_.astype(str)))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_nn)))

print("✅ Combined results saved")

# =====================================
# GRAPHS
# =====================================
def plot_cm(cm, title, filename, cmap):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

plot_cm(confusion_matrix(y_test, y_pred_rf), "RF Confusion Matrix", "../result/wsn_ds_rf_confusion_multi.png", "Reds")
plot_cm(confusion_matrix(y_test, y_pred_svm), "SVM Confusion Matrix", "../result/wsn_ds_svm_confusion_multi.png", "Blues")
plot_cm(confusion_matrix(y_test, y_pred_nn), "DNN Confusion Matrix", "../result/wsn_ds_dnn_confusion_multi.png", "Greens")

models = ['RF', 'SVM', 'DNN']
accuracies = [rf_accuracy, svm_accuracy, dnn_accuracy]
plt.figure(figsize=(6,5))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("WSN-DS MULTI Accuracy Comparison")
plt.savefig("../result/wsn_ds_accuracy_comparison_multi.png")
plt.close()

print("✅ Graphs saved")
print("\n🎯 WSN-DS MULTI EXPERIMENT COMPLETED!")
