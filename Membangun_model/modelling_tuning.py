# Import library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# Setup DagsHub — ganti sesuai username dan nama repo DagsHub kamu
dagshub.init(repo_owner='Escavora', 
             repo_name='Eksperimen_SML_Alief_Athallah', 
             mlflow=True)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'stroke_preprocessing', 'stroke_preprocessed.csv'))

print(f"Dataset loaded: {df.shape}")

# Split features dan target
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

print("\nMemulai GridSearchCV...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# MLflow manual logging
mlflow.set_experiment("stroke-prediction-tuning")

with mlflow.start_run(run_name="RandomForest-GridSearchCV"):

    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Manual log metrics (sama seperti autolog)
    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("roc_auc",   roc_auc_score(y_test, y_prob))

    # Artefak tambahan 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['No Stroke', 'Stroke']).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Best Model")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")
    print("Artefak 1 (confusion_matrix.png) di-log.")

    # Artefak tambahan 2: Feature Importance
    features = X.columns.tolist()
    importances = best_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(features)), importances[idx], color='steelblue', edgecolor='white')
    plt.xticks(range(len(features)), [features[i] for i in idx], rotation=45, ha='right')
    plt.title("Feature Importance — Best Model")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()
    mlflow.log_artifact("feature_importance.png")
    print("Artefak 2 (feature_importance.png) di-log.")

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    print(f"\nBest params : {grid_search.best_params_}")
    print(f"Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score    : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC     : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nSemua metrics dan artefak berhasil di-log ke DagsHub.")