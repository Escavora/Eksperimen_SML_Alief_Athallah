# preprocessing/automate_Alief_Athallah.py

# Import library
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import os

# Path otomatis agar bekerja di local & GitHub Actions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_file      = os.path.join(BASE_DIR, "dataset-stroke-data.csv")
output_folder = os.path.join(BASE_DIR, "preprocessing", "stroke_preprocessing")
output_file   = os.path.join(output_folder, "stroke_preprocessed.csv")

os.makedirs(output_folder, exist_ok=True)

# Load Dataset
df = pd.read_csv(raw_file)
print("Dataset berhasil dibaca, contoh data:")
print(df.head())
print(f"\nShape awal: {df.shape}")

# Drop kolom tidak relevan (id tidak berguna untuk training)
df = df.drop(columns=['id'], errors='ignore')

# Menghapus Duplikat
df = df.drop_duplicates()
print(f"Shape setelah drop id + duplikat: {df.shape}")

# Menentukan kolom numerik, kategorikal, dan target
TARGET_COL      = 'stroke'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = [c for c in df.select_dtypes(include=['number']).columns
                    if c != TARGET_COL]   # exclude target dari scaling

print(f"\nKolom numerik  : {numerical_cols}")
print(f"Kolom kategorikal: {categorical_cols}")

# Handle Missing Values (kolom 'bmi' memiliki ~201 nilai kosong)
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
print(f"\nMissing values tersisa: {df.isnull().sum().sum()}")

# Simpan Imputer
imputer_path = os.path.join(output_folder, "imputer.pkl")
joblib.dump(imputer, imputer_path)
print(f"Imputer disimpan di: {imputer_path}")

# Encoding LabelEncoder Kolom Kategorikal
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Simpan Label Encoders
label_encoder_path = os.path.join(output_folder, "label_encoders.pkl")
joblib.dump(label_encoders, label_encoder_path)
print(f"Label Encoders disimpan di: {label_encoder_path}")

# Scaling kolom Numerik dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ],
    remainder='drop'
)

# Menjalankan preprocessing scaling
data_scaled = preprocessor.fit_transform(df[numerical_cols])

# Simpan Preprocessor (scaler)
preprocessor_path = os.path.join(output_folder, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor (scaler) disimpan di: {preprocessor_path}")

# Menggabungkan kolom numerik (scaled) dan kategorikal (encoded) + target
df_preprocessed = df.copy()
df_preprocessed[numerical_cols] = data_scaled

# Simpan hasil dalam format csv
df_preprocessed.to_csv(output_file, index=False)
print(f"\nHasil preprocessing disimpan di: {output_file}")
print(f"Shape akhir: {df_preprocessed.shape}")
print("\nContoh data preprocessed:")
print(df_preprocessed.head())