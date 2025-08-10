# scripts/fraud_pipeline.py
# ============================================================
# Detección de Fraude Transaccional con Ensembles + AutoML
# - Carga CSV real o simula dataset
# - Preprocesado (num/cat), feature engineering
# - Modelos: LogisticRegression, RandomForest, XGBoost
# - Selección por AUC-PR con RandomizedSearchCV
# - Calibración de probabilidades
# - Umbral óptimo por valor de negocio (coste fraude vs coste revisión)
# - Exporta: modelo, preprocesador, métricas, gráficos y reporte
# ============================================================

import os
import sys
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# XGBoost (opcional pero recomendado)
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

# SHAP (opcional)
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------
# Data loading / simulation
# ------------------------------------------------------------
def load_or_simulate(data_dir: str, n_simulate: int = 30000) -> pd.DataFrame:
    """
    Espera data/transactions.csv con:
      - label (0/1), amount, user_id, merchant_id, country, device, category, ts (ISO)
      - opcionales: zipcode, hour, ...
    Si no existe, simula uno realista.
    """
    ensure_dir(data_dir)
    csv_path = os.path.join(data_dir, "transactions.csv")
    if os.path.exists(csv_path):
        logging.info(f"Leyendo {csv_path}")
        df = pd.read_csv(csv_path)
        # normaliza timestamps si existen
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    logging.info("No hay CSV, simulando dataset realista...")
    n = int(n_simulate)

    # Usuarios y merchants
    n_users = max(1000, n // 15)
    n_merch = max(300, n // 50)

    df = pd.DataFrame({
        "tx_id": np.arange(1, n + 1),
        "user_id": np.random.randint(1, n_users + 1, size=n),
        "merchant_id": np.random.randint(1, n_merch + 1, size=n),
        "amount": np.round(np.random.gamma(shape=2.2, scale=35, size=n) + 3, 2),
        "country": np.random.choice(["ES", "FR", "DE", "IT", "PT", "GB", "US"], size=n, p=[.35,.12,.12,.14,.09,.1,.08]),
        "device": np.random.choice(["android", "ios", "web"], size=n, p=[.48,.32,.20]),
        "category": np.random.choice(["electronics","fashion","grocery","gaming","travel","services"], size=n),
    })
    # tiempo
    start = pd.Timestamp("2024-01-01")
    df["ts"] = start + pd.to_timedelta(np.random.randint(0, 180, size=n), unit="D") \
                        + pd.to_timedelta(np.random.randint(0, 24*60, size=n), unit="m")

    # Patrón de fraude (probabilidades condicionales)
    base = 0.012  # 1.2% base fraud
    risk_country = df["country"].isin(["GB","US"]).astype(int) * 0.010
    risk_night = ((df["ts"].dt.hour < 6) | (df["ts"].dt.hour > 22)).astype(int) * 0.008
    risk_high_amount = (df["amount"] > df["amount"].quantile(0.90)).astype(int) * 0.020
    risk_device = (df["device"] == "web").astype(int) * 0.006
    risk_cat = df["category"].isin(["electronics","gaming","travel"]).astype(int) * 0.007

    p_fraud = base + risk_country + risk_night + risk_high_amount + risk_device + risk_cat
    df["label"] = (np.random.rand(n) < p_fraud).astype(int)

    # Un poco de “concept drift” suave a final de periodo
    late = (df["ts"] > (start + pd.Timedelta(days=120))).astype(int)
    df.loc[late.astype(bool), "label"] = ((np.random.rand(late.sum()) < (p_fraud[late.astype(bool)] + 0.005))).astype(int)

    return df


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Extraer variables temporales
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Ratios y transformaciones
    df["log_amount"] = np.log1p(df["amount"])

    # Frecuencias por usuario y merchant (aprox, sin leakage fuerte)
    # Para evitar fuga temporal severa, se puede calcular en train y mapear a test; aquí simplificado.
    user_freq = df["user_id"].value_counts().to_dict()
    merch_freq = df["merchant_id"].value_counts().to_dict()
    df["user_tx_freq"] = df["user_id"].map(user_freq).fillna(1)
    df["merchant_tx_freq"] = df["merchant_id"].map(merch_freq).fillna(1)

    # Ticket medio por merchant (aprox)
    df["merchant_amount_mean"] = df.groupby("merchant_id")["amount"].transform("mean")

    # Variables de interacción simples
    df["amount_over_merchant_mean"] = df["amount"] / (df["merchant_amount_mean"] + 1e-6)

    return df


def train_valid_test_split(df: pd.DataFrame, test_days: int = 14, valid_days: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal para evitar leakage."""
    df = df.sort_values("ts")
    tmax = df["ts"].max()
    test_cut = tmax - pd.Timedelta(days=test_days)
    valid_cut = test_cut - pd.Timedelta(days=valid_days)
    train = df[df["ts"] <= valid_cut]
    valid = df[(df["ts"] > valid_cut) & (df["ts"] <= test_cut)]
    test = df[df["ts"] > test_cut]
    return train, valid, test


# ------------------------------------------------------------
# Modeling
# ------------------------------------------------------------
def build_preprocessor(train: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    num_cols = ["amount","log_amount","hour","dayofweek","is_night","is_weekend",
                "user_tx_freq","merchant_tx_freq","merchant_amount_mean","amount_over_merchant_mean"]
    cat_cols = ["country","device","category"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50 if len(train) > 20000 else 5), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


def candidate_models(class_weight: Dict[int, float]):
    models = []

    # 1) Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight=class_weight, n_jobs=None)
    lr_params = {
        "clf__C": np.logspace(-3, 2, 30),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    models.append(("LogReg", lr, lr_params))

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, class_weight=class_weight
    )
    rf_params = {
        "clf__n_estimators": [300, 400, 600],
        "clf__max_depth": [8, 12, 16, None],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.5, 0.8]
    }
    models.append(("RandomForest", rf, rf_params))

    # 3) XGBoost (si está disponible)
    if XGB_OK:
        xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=600,
            n_jobs=-1,
            # scale_pos_weight se ajusta después
        )
        models.append((
            "XGBoost",
            xgb,
            {
                "clf__max_depth": [3, 4, 6, 8],
                "clf__learning_rate": [0.03, 0.05, 0.08],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
                "clf__min_child_weight": [1, 3, 5],
                "clf__gamma": [0, 0.5, 1.0],
            }
        ))
    return models


def fit_and_select_model(pre: ColumnTransformer,
                         X_tr: pd.DataFrame, y_tr: np.ndarray,
                         X_va: pd.DataFrame, y_va: np.ndarray):
    """Entrena varios modelos, selecciona por AUC-PR en valid."""
    # pesos por clase
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    best = {"name": None, "estimator": None, "ap": -np.inf, "roc": -np.inf, "probs": None}

    for name, clf, params in candidate_models(class_weight):
        logging.info(f"Buscando hiperparámetros: {name}")

        # Ajuste especial para XGBoost: scale_pos_weight
        if XGB_OK and isinstance(clf, XGBClassifier):
            # ratio negativos/positivos
            ratio = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
            clf.set_params(scale_pos_weight=ratio)

        pipe = Pipeline([("pre", pre), ("clf", clf)])
        rcv = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=25 if name != "LogReg" else 15,
            scoring="average_precision",
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        rcv.fit(X_tr, y_tr)
        probs = rcv.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, probs)
        roc = roc_auc_score(y_va, probs)
        logging.info(f"{name} | AUC-PR={ap:.4f} | ROC-AUC={roc:.4f}")

        if ap > best["ap"]:
            best = {"name": name, "estimator": rcv.best_estimator_, "ap": ap, "roc": roc, "probs": probs}

    logging.info(f"Mejor modelo: {best['name']} (AUC-PR={best['ap']:.4f})")
    return best


# ------------------------------------------------------------
# Threshold optimization by business value
# ------------------------------------------------------------
def optimize_threshold(y_true: np.ndarray, p: np.ndarray, cost_fraud: float, cost_review: float) -> Dict[str, Any]:
    """
    Devuelve el umbral que maximiza el valor esperado:
      Valor = TP * cost_fraud (ahorro) - FP * cost_review (coste de revisar)
    (equivale a minimizar coste esperado)
    """
    prec, rec, th = precision_recall_curve(y_true, p)
    # th tiene len = n-1, generamos vectores alineados
    thresholds = np.r_[0.0, th, 1.0]
    values = []
    best_idx = 0

    for i, t in enumerate(thresholds):
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        value = tp * cost_fraud - fp * cost_review
        values.append(value)
        if value >= values[best_idx]:
            best_idx = i

    return {
        "threshold": float(thresholds[best_idx]),
        "expected_value": float(values[best_idx]),
        "pr_points": int(len(thresholds)),
    }


# ------------------------------------------------------------
# Feature importance / SHAP
# ------------------------------------------------------------
def export_importances(model: Pipeline, feature_names: list, out_csv: str, out_png: str):
    try:
        clf = model.named_steps["clf"]
        # RandomForest o XGBoost
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
            df_imp.sort_values("importance", ascending=False, inplace=True)
            df_imp.to_csv(out_csv, index=False)

            plt.figure(figsize=(8, 10))
            top = df_imp.head(25)
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Top Importancias (árboles)")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
    except Exception:
        pass


def export_shap_summary(model: Pipeline, X_sample: pd.DataFrame, out_png: str):
    if not SHAP_OK:
        return
    try:
        clf = model.named_steps["clf"]
        pre = model.named_steps["pre"]
        X_enc = pre.transform(X_sample)
        explainer = None
        if XGB_OK and isinstance(clf, XGBClassifier):
            explainer = shap.TreeExplainer(clf)
        elif hasattr(clf, "estimators_"):  # RandomForest
            explainer = shap.TreeExplainer(clf)
        else:
            return
        shap_values = explainer.shap_values(X_enc)
        plt.figure()
        shap.summary_plot(shap_values, X_enc, show=False, plot_size=(8, 6))
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception:
        pass


# ------------------------------------------------------------
# Evaluation report
# ------------------------------------------------------------
def export_plots_and_report(outdir: str, name: str,
                            y_true_va: np.ndarray, p_va: np.ndarray,
                            y_true_te: np.ndarray, p_te: np.ndarray,
                            best_threshold: float, metrics: Dict[str, Any],
                            feature_names: list):
    # Curvas PR y ROC en validation
    prec, rec, th = precision_recall_curve(y_true_va, p_va)
    fpr, tpr, _ = roc_curve(y_true_va, p_va)

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (valid) – {name}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "pr_valid.png"), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (valid) – {name}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "roc_valid.png"), dpi=150); plt.close()

    # Confusion en test con umbral elegido
    yhat_te = (p_te >= best_threshold).astype(int)
    cm = confusion_matrix(y_true_te, yhat_te)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Fraude","Fraude"]); ax.set_yticklabels(["No Fraude","Fraude"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Matriz de confusión (test)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_test.png"), dpi=150); plt.close()

    # Reporte markdown
    md = []
    md.append(f"# Fraud Detection – Reporte Ejecutivo\n")
    md.append(f"- Modelo seleccionado: **{name}**\n")
    md.append(f"- AUC-PR (valid): **{metrics['ap_valid']:.4f}** · ROC-AUC (valid): **{metrics['roc_valid']:.4f}**\n")
    md.append(f"- AUC-PR (test): **{metrics['ap_test']:.4f}** · ROC-AUC (test): **{metrics['roc_test']:.4f}**\n")
    md.append(f"- Umbral óptimo por valor de negocio: **{metrics['best_threshold']:.4f}**\n")
    md.append(f"- Valor esperado por 10k transacciones (aprox.): **{metrics['expected_value_per10k']:.2f} €**\n")
    md.append(f"\n## Gráficos\n")
    md.append(f"- Precision-Recall (valid): `pr_valid.png`\n")
    md.append(f"- ROC (valid): `roc_valid.png`\n")
    md.append(f"- Matriz de confusión (test): `confusion_test.png`\n")
    md.append(f"- Importancias de features: `feature_importance.png` (si aplica)\n")
    if SHAP_OK:
        md.append(f"- SHAP summary (muestra): `shap_summary.png`\n")

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--data-dir", default="data", help="Carpeta de datos (transactions.csv) o vacía para simulación")
    parser.add_argument("--outdir", default="artifacts", help="Carpeta de salida de artefactos")
    parser.add_argument("--simulate-n", type=int, default=30000, help="Tamaño del dataset simulado si no hay CSV")
    parser.add_argument("--cost-fraud", type=float, default=200.0, help="€ perdidos si un fraude NO se detecta (FN)")
    parser.add_argument("--cost-review", type=float, default=3.0, help="€ por revisar un FP (costo operativo)")
    args = parser.parse_args()

    t0 = time.time()
    ensure_dir(args.outdir)

    # 1) Cargar datos
    df_raw = load_or_simulate(args.data_dir, n_simulate=args.simulate_n)

    # 2) Features + split temporal
    df = build_features(df_raw)
    train, valid, test = train_valid_test_split(df, test_days=14, valid_days=14)
    logging.info(f"Split -> train={len(train)}, valid={len(valid)}, test={len(test)}")

    target = "label"
    feats = [c for c in df.columns if c not in [target, "tx_id", "ts"]]
    X_tr, y_tr = train[feats], train[target].values
    X_va, y_va = valid[feats], valid[target].values
    X_te, y_te = test[feats], test[target].values

    # 3) Preprocesador
    pre, num_cols, cat_cols = build_preprocessor(train)

    # 4) Búsqueda de modelos y selección
    best = fit_and_select_model(pre, X_tr, y_tr, X_va, y_va)
    best_model = best["estimator"]
    p_va = best["probs"]
    ap_va = average_precision_score(y_va, p_va)
    roc_va = roc_auc_score(y_va, p_va)

    # 5) Calibración en train+valid y evaluación en test
    logging.info("Calibrando probabilidades (sigmoid)...")
    X_trva = pd.concat([X_tr, X_va], axis=0)
    y_trva = np.concatenate([y_tr, y_va], axis=0)

    cal = CalibratedClassifierCV(best_model, method="sigmoid", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE))
    cal.fit(X_trva, y_trva)
    p_te = cal.predict_proba(X_te)[:, 1]
    ap_te = average_precision_score(y_te, p_te)
    roc_te = roc_auc_score(y_te, p_te)
    logging.info(f"Test | AUC-PR={ap_te:.4f} | ROC-AUC={roc_te:.4f}")

    # 6) Umbral por valor de negocio (optimizado en VALID)
    best_thr_info = optimize_threshold(y_va, p_va, cost_fraud=args.cost_fraud, cost_review=args.cost_review)
    thr = best_thr_info["threshold"]
    value = best_thr_info["expected_value"]

    # 7) Export importancias y SHAP
    # Nombres de features “post-encoding”
    ohe = best_model.named_steps["pre"].named_transformers_["cat"]
    num_names = num_cols
    cat_names = list(ohe.get_feature_names_out(["country","device","category"]))
    feature_names = num_names + cat_names
    export_importances(
        model=best_model,
        feature_names=feature_names,
        out_csv=os.path.join(args.outdir, "feature_importance.csv"),
        out_png=os.path.join(args.outdir, "feature_importance.png")
    )
    if SHAP_OK:
        sample = X_te.sample(min(2000, len(X_te)), random_state=RANDOM_STATE)
        export_shap_summary(best_model, sample, os.path.join(args.outdir, "shap_summary.png"))

    # 8) Guardar artefactos (modelo calibrado y preprocesador de la versión final)
    # Para deploy real convendría re-ajustar el preprocesador completo; aquí guardamos el calibrado completo como pipeline
    dump(cal, os.path.join(args.outdir, "fraud_model_calibrated.joblib"))

    # 9) Métricas y reporte
    metrics = {
        "best_model": best["name"],
        "ap_valid": float(ap_va),
        "roc_valid": float(roc_va),
        "ap_test": float(ap_te),
        "roc_test": float(roc_te),
        "best_threshold": float(thr),
        "expected_value_valid": float(value),
        "expected_value_per10k": float(value * (10000 / max(1, len(valid)))),
        "n_train": int(len(train)), "n_valid": int(len(valid)), "n_test": int(len(test)),
        "fraud_rate_train": float(train[target].mean()),
        "fraud_rate_valid": float(valid[target].mean()),
        "fraud_rate_test": float(test[target].mean()),
    }
    save_json(metrics, os.path.join(args.outdir, "metrics.json"))

    export_plots_and_report(
        outdir=args.outdir, name=best["name"],
        y_true_va=y_va, p_va=p_va,
        y_true_te=y_te, p_te=p_te,
        best_threshold=thr, metrics=metrics,
        feature_names=feature_names
    )

    # 10) Imprimir resumen ejecutivo
    logging.info("=== Resumen Ejecutivo ===")
    logging.info(json.dumps(metrics, indent=2))
    logging.info(f"Artefactos guardados en: {os.path.abspath(args.outdir)}")
    logging.info(f"Duración total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
