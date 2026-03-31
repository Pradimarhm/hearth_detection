# ============================================
# Sistem Pakar + ANN Hybrid Prediction
# ============================================

import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from flask import Flask, render_template 

# Load model & scaler
model = load_model("best_model.keras")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")




# ============================================
# 1. Fungsi Sistem Pakar
# ============================================
def rule_based_system(data):
    score = 0
    if data['Blood Pressure'] > 140:
        score += 1
    if data['Cholesterol Level'] > 200:
        score += 1
    if data['Smoking'] == 1:
        score += 1
    if data.get('Exercise Habits', 0) == 0:
        score += 1

    return 1 if score >= 2 else 0  # Aturan sederhana

# ============================================
# Fungsi Konversi Input Teks ke Angka
# ============================================
def convert_input(data):
    mapping = {
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0,
        'Low': 0, 'Medium': 1, 'High': 2
    }

    converted = {}
    for key, value in data.items():
        # Jika value ada di mapping, ubah ke angka
        if isinstance(value, str) and value in mapping:
            converted[key] = mapping[value]
        else:
            converted[key] = value  # biarkan angka tetap angka
    return converted

# ============================================
# 2. Fungsi Hybrid Prediction
# ============================================
def hybrid_predict(data_input):
    # Konversi input teks ke angka
    data_input = convert_input(data_input)
    
    # Sistem pakar
    rule_result = rule_based_system(data_input)
    
    # Pastikan nama kolom sama
    # X_input = pd.DataFrame([data_input])

    # # Pastikan kolom sesuai urutan training
    # for col in feature_names:
    #     if col not in X_input.columns:
    #         X_input[col] = 0  # tambahkan kolom yang hilang
    # X_input = X_input[feature_names]  # urutkan sesuai training

    # X_scaled = scaler.transform(X_input)

    # Prediksi model ANN
    X_input = pd.DataFrame([data_input])
    
    # Pastikan semua kolom yang dilatih scaler ada di input
    for col in feature_names:
        if col not in X_input.columns:
            X_input[col] = 0  # tambahkan kolom kosong kalau hilang

    # Urutkan kolom sesuai urutan training
    X_input = X_input[feature_names]
    
    X_scaled = scaler.transform(X_input)
    ann_pred = model.predict(X_scaled)[0][0]

    # Gabungan hasil
    final_score = (rule_result + ann_pred) / 2
    diagnosis = "Tinggi" if final_score > 0.5 else "Rendah"

    return {
        "rule_result": int(rule_result),
        "ann_prediction": float(ann_pred),
        "final_score": float(final_score),
        "diagnosis": diagnosis
    }

# ============================================
# 3. Uji Coba
# ============================================
# if __name__ == "__main__":
#     sample_data = {
#         'Age': 55,
#         'Gender': 'Male',
#         'Blood Pressure': 120,
#         'Cholesterol Level': 180,
#         'Smoking': 'No',
#         'Family Heart Disease': 'No',
#         'Exercise Habits': 'High',
#         'Alcohol Consumption': 'Medium',
#         'BMI': 27,
#         'CRP Level': 12,
#         'Diabetes': 'No',
#         'High Blood Pressure': 'Yes',
#         'Low HDL Cholesterol': 'No',
#         'High LDL Cholesterol': 'Yes',
#         'Stress Level': 'Medium',
#         'Sleep Hours': 6,
#         'Sugar Consumption': 'Low',
#         'Triglyceride Level': 230,
#         'Fasting Blood Sugar': 120,
#         'CRP Level': 11,
#         'Homocysteine Level': 9
#         # Tambahkan kolom lain sesuai model
#     }
    
    

#     hasil = hybrid_predict(sample_data)

    # print("\n=== HASIL DIAGNOSIS HYBRID ===")
    # print("Rule-based:", hasil['rule_result'])
    # print("Prediksi ANN:", round(hasil['ann_prediction'], 3))
    # print("Final Score:", round(hasil['final_score'], 3))
    # print("Kesimpulan:", hasil['diagnosis'])
    
    
