# ============================================
# Prediksi Penyakit Jantung dengan ANN + Visualisasi
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv("../heart_dl/heart_disease/heart_disease.csv")

print(df.columns.tolist())


print("\n=== INFO DATA ===")
print(df.info())
print(df.head())

# ============================================
# 2️ HANDLE MISSING VALUE
# ============================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ============================================
# 3️ ENCODING FITUR KATEGORIKAL
# ============================================
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Heart Disease Status')  # target jangan di-encode

# df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df['Heart Disease Status'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
df = df.replace({       
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0,
    'Low': 0, 'Medium': 1, 'High': 2
})
# df['Sugar_Consumption'] = df['Sugar_Consumption'].map({
#     'Low': 0,
#     'Medium': 1,
#     'High': 2
# })

# ============================================
# 4️ SPLIT FITUR DAN TARGET
# ============================================
X = df.drop('Heart Disease Status', axis=1)
y = df['Heart Disease Status']

# ============================================
# 5️ STANDARISASI & BALANCING DATA
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# ============================================
# 6️ TRAIN TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)

# ============================================
# 7️ BANGUN MODEL ANN
# ============================================
model = Sequential([
    Dense(640, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(320, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ============================================
# 8️ TRAINING MODEL
# ============================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

history = model.fit(
    X_train, y_train, 
    # class_weight='balanced',
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ============================================
# 9️ VISUALISASI HASIL TRAINING
# ============================================
plt.figure(figsize=(12, 5))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 🔟 EVALUASI MODEL
# ============================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # flatten supaya 1D

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Class 0', 'Class 1']

# Hitung persentase tiap sel
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Buat anotasi gabungan angka + persentase
annot_labels = np.array([["{} ({:.1%})".format(cm[i,j], cm_percent[i,j]) 
                            for j in range(cm.shape[1])] 
                            for i in range(cm.shape[0])])

plt.figure(figsize=(6,5))
sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap dengan Persentase')
plt.show()

# ============================================
# SIMPAN MODEL DAN SCALER
# ============================================
model.save("best_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Model dan scaler berhasil disimpan!")