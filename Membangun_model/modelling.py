# Import library
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup DagsHub — ganti sesuai username dan nama repo DagsHub kamu
dagshub.init(repo_owner='Escavora', 
             repo_name='Eksperimen_SML_Alief_Athallah', 
             mlflow=True)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'stroke_preprocessing', 'stroke_preprocessed.csv'))

print(f"Dataset loaded: {df.shape}")
print(df.head())

# Split features dan target
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training dengan MLflow autolog
mlflow.set_experiment("stroke-prediction-modelling")

with mlflow.start_run(run_name="RandomForest-autolog"):
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\nModel accuracy: {score:.4f}")
    print("Run berhasil di-log ke DagsHub MLflow.")