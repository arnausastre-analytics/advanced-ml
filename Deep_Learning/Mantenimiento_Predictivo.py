# scripts/anomaly_detection_pipeline.py
# ==============================================================
# Detección y Clasificación de Anomalías Multivariante
# - Compatible con datos CSV o streaming simulado
# - Preprocesado avanzado con escalado y PCA opcional
# - Modelos: IsolationForest, Autoencoder
# - Clasificación de anomalías con clustering y reglas
# - Explicabilidad con SHAP
# - Exporta métricas, gráficos y reporte
# ==============================================================

import os
import json
import time
import argparse
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# Deep learning
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except Exception:
    TF_OK = False

# SHAP explicabilidad
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------- Utils --------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(data: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------- Simulación de datos --------------------
def simulate_data(n_samples=5000, n_features=10, anomaly_frac=0.05) -> pd.DataFrame:
    logging.info(f"Simulando dataset con {n_samples} registros y {n_features} variables...")
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    n_anomalies = int(n_samples * anomaly_frac)

    # Introducir anomalías
    anomalies = np.random.normal(5, 1.5, size=(n_anomalies, n_features))
    X[:n_anomalies] = anomalies

    cols = [f"var_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["is_anomaly"] = 0
    df.iloc[:n_anomalies, -1] = 1
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# -------------------- Modelos --------------------
def train_isolation_forest(X_train):
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=RANDOM_STATE
    )
    model.fit(X_train)
    return model

def train_autoencoder(X_train, encoding_dim=5):
    if not TF_OK:
        logging.warning("TensorFlow no está disponible, se omite Autoencoder.")
        return None
    input_dim = X_train.shape[1]
    model = Sequential([
        Dense(encoding_dim, activation="relu", input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(input_dim, activation="linear")
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, X_train, epochs=50, batch_size=64, verbose=0, callbacks=[es])
    return model


# -------------------- Clasificación de anomalías --------------------
def classify_anomalies(X_anomalies, n_clusters=3):
    if len(X_anomalies) < n_clusters:
        return np.zeros(len(X_anomalies))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    return km.fit_predict(X_anomalies)


# -------------------- Explicabilidad --------------------
def explain_with_shap(model, X_sample, out_path):
    if not SHAP_OK:
        logging.warning("SHAP no disponible, se omite explicación.")
        return
    try:
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"No se pudo generar explicabilidad SHAP: {e}")


# -------------------- Pipeline principal --------------------
def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument("--data", default="", help="Ruta a CSV con datos, o vacío para simular")
    parser.add_argument("--outdir", default="artifacts_anomalies", help="Carpeta de salida")
    parser.add_argument("--pca", action="store_true", help="Aplicar PCA")
    parser.add_argument("--n_features", type=int, default=10, help="Número de variables si se simula")
    parser.add_argument("--n_samples", type=int, default=5000, help="Muestras si se simula")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # 1) Cargar o simular datos
    if args.data and os.path.exists(args.data):
        logging.info(f"Leyendo datos desde {args.data}")
        df = pd.read_csv(args.data)
    else:
        df = simulate_data(args.n_samples, args.n_features)

    X = df.drop(columns=["is_anomaly"])
    y = df["is_anomaly"].values

    # 2) Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) PCA opcional
    if args.pca:
        pca = PCA(n_components=min(5, X.shape[1]))
        X_scaled = pca.fit_transform(X_scaled)
        logging.info(f"PCA aplicado: {X_scaled.shape[1]} componentes.")

    # 4) Entrenar Isolation Forest
    iso = train_isolation_forest(X_scaled)
    iso_pred = (iso.predict(X_scaled) == -1).astype(int)

    # 5) Entrenar Autoencoder (opcional)
    ae_pred = np.zeros_like(y)
    if TF_OK:
        ae = train_autoencoder(X_scaled)
        recon_error = np.mean(np.square(X_scaled - ae.predict(X_scaled)), axis=1)
        threshold = np.percentile(recon_error, 95)
        ae_pred = (recon_error > threshold).astype(int)

    # 6) Unir predicciones
    final_pred = np.maximum(iso_pred, ae_pred)

    # 7) Clasificar anomalías
    anomaly_idx = np.where(final_pred == 1)[0]
    anomaly_clusters = classify_anomalies(X_scaled[anomaly_idx]) if len(anomaly_idx) else []

    # 8) Métricas
    report = classification_report(y, final_pred, output_dict=True)
    save_json(report, os.path.join(args.outdir, "metrics.json"))

    # 9) Plots
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(y)), np.zeros_like(y), c=final_pred, cmap="coolwarm", alpha=0.6)
    plt.title("Predicciones de anomalías")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Anomalía (0/1)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "anomaly_predictions.png"), dpi=150)
    plt.close()

    # 10) Explicabilidad (solo para Isolation Forest en SHAP)
    explain_with_shap(iso, X_scaled[:200], os.path.join(args.outdir, "shap_summary.png"))

    # 11) Guardar predicciones
    df_out = df.copy()
    df_out["pred_anomaly"] = final_pred
    if len(anomaly_idx):
        df_out.loc[anomaly_idx, "anomaly_cluster"] = anomaly_clusters
    df_out.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    logging.info("Pipeline completado.")
    logging.info(f"Artefactos guardados en: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()

