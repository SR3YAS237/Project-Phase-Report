import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

    # ~300k rows
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
    lambda x: 0 if x == 'normal.' else 1
)

# =====================================
# ENCODE CATEGORICAL FEATURES
# =====================================
df = pd.get_dummies(
    df,
    columns=['protocol_type', 'service', 'flag']
)

# =====================================
# FEATURES AND LABELS
# =====================================
X = df.drop('label', axis=1)

y = df['label']

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
# TRAIN SVM
# =====================================
print("\nTraining Binary SVM...")

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================
y_pred = svm_model.predict(X_test)

# =====================================
# METRICS
# =====================================
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n===== BINARY SVM RESULTS =====")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

# =====================================
# SAVE RESULTS
# =====================================
with open("svm_kdd_binary.txt", "w") as f:

    f.write("===== BINARY SVM =====\n\n")

    f.write(f"Accuracy : {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall   : {recall}\n")
    f.write(f"F1 Score : {f1}\n\n")

    f.write("Classification Report:\n")

    f.write(
        classification_report(
            y_test,
            y_pred
        )
    )

    f.write("\nConfusion Matrix:\n")

    f.write(
        str(
            confusion_matrix(
                y_test,
                y_pred
            )
        )
    )

print("✅ Binary SVM results saved")

# =====================================
# CONFUSION MATRIX GRAPH
# =====================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title("Binary SVM Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("svm_binary_confusion.png")

plt.close()

print("✅ Confusion matrix graph saved")

print("\n🎯 BINARY SVM EXPERIMENT COMPLETED!")