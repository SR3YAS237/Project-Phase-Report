import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

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
# MULTI-CLASS LABEL MAPPING
# =====================================
def map_attack(label):

    if label == 'normal.':
        return 'normal'

    dos = [
        'back.', 'land.', 'neptune.',
        'pod.', 'smurf.', 'teardrop.'
    ]

    probe = [
        'ipsweep.', 'nmap.',
        'portsweep.', 'satan.'
    ]

    r2l = [
        'ftp_write.', 'guess_passwd.',
        'imap.', 'multihop.',
        'phf.', 'spy.',
        'warezclient.', 'warezmaster.'
    ]

    u2r = [
        'buffer_overflow.',
        'loadmodule.',
        'perl.',
        'rootkit.'
    ]

    if label in dos:
        return 'DoS'

    elif label in probe:
        return 'Probe'

    elif label in r2l:
        return 'R2L'

    elif label in u2r:
        return 'U2R'

    else:
        return 'other'

df['label'] = df['label'].apply(map_attack)

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
# FULL FEATURES
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
# CLASS WEIGHTS
# =====================================
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = {
    i: min(w, 50)
    for i, w in enumerate(weights)
}

# =====================================
# SVM
# =====================================
print("\nTraining SVM...")

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM Accuracy:", svm_accuracy)

with open("svm_kdd_multi.txt", "w") as f:

    f.write("===== SVM =====\n\n")

    f.write(
        classification_report(
            y_test,
            y_pred_svm,
            labels=np.unique(y_test),
            target_names=le.classes_[np.unique(y_test)]
        )
    )

    f.write("\nConfusion Matrix:\n")

    f.write(
        str(
            confusion_matrix(
                y_test,
                y_pred_svm
            )
        )
    )

    f.write(
        f"\n\nAccuracy: "
        f"{svm_accuracy}"
    )

print("✅ SVM results saved")

# =====================================
# DNN
# =====================================
print("\nTraining DNN...")

num_classes = len(np.unique(y_train))

model_nn = Sequential([
    Input(shape=(X_train.shape[1],)),

    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
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
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

y_pred_nn = np.argmax(
    model_nn.predict(X_test),
    axis=1
)

dnn_accuracy = accuracy_score(y_test, y_pred_nn)

print("\nDNN Accuracy:", dnn_accuracy)

with open("dnn_kdd_multi.txt", "w") as f:

    f.write("===== DNN =====\n\n")

    f.write(
        classification_report(
            y_test,
            y_pred_nn,
            labels=np.unique(y_test),
            target_names=le.classes_[np.unique(y_test)]
        )
    )

    f.write("\nConfusion Matrix:\n")

    f.write(
        str(
            confusion_matrix(
                y_test,
                y_pred_nn
            )
        )
    )

    f.write(
        f"\n\nAccuracy: "
        f"{dnn_accuracy}"
    )

print("✅ DNN results saved")

# =====================================
# SVM CONFUSION MATRIX GRAPH
# =====================================
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_[np.unique(y_test)],
    yticklabels=le.classes_[np.unique(y_test)]
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("svm_confusion.png")

plt.close()

# =====================================
# DNN CONFUSION MATRIX GRAPH
# =====================================
cm_dnn = confusion_matrix(y_test, y_pred_nn)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_dnn,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=le.classes_[np.unique(y_test)],
    yticklabels=le.classes_[np.unique(y_test)]
)

plt.title("DNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("dnn_confusion.png")

plt.close()

# =====================================
# ACCURACY COMPARISON GRAPH
# =====================================
models = ['SVM', 'DNN']

accuracies = [
    svm_accuracy,
    dnn_accuracy
]

plt.figure(figsize=(5,5))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.savefig("accuracy_comparison.png")

plt.close()

print("✅ Graphs saved")

print("\n🎯 EXPERIMENT COMPLETED!")