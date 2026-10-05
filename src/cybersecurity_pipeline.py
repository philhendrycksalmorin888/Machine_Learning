"""Reusable pipeline for the Machine Learning Cybersecurity Project.

Expected input columns:
timestamp, src_ip, dst_ip, src_port, dst_port, protocol, bytes_sent,
bytes_received, user_agent, url, is_internal_traffic, label, attack_type
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, classification_report
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
F2_THRESHOLD = 0.2286

REQUIRED_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "bytes_sent", "bytes_received", "user_agent", "url", "is_internal_traffic",
    "label", "attack_type"
]

NUMERIC_FEATURES = [
    "src_port", "dst_port", "log_bytes_sent", "log_bytes_received",
    "is_internal_traffic", "hour", "day_of_week", "url_present"
]
CATEGORICAL_FEATURES = ["protocol", "user_agent", "url_path"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def url_path(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "(missing)"
    try:
        parsed = urlparse(str(value))
        path = parsed.path or "(root)"
        return path.lower()
    except Exception:
        return "(invalid)"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out[out["label"].isin([0, 1])].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["url_present"] = (~out["url"].isna() & (out["url"].astype(str).str.strip() != "")).astype(int)
    out["url_path"] = out["url"].apply(url_path)
    out["log_bytes_sent"] = np.log1p(pd.to_numeric(out["bytes_sent"], errors="coerce").clip(lower=0))
    out["log_bytes_received"] = np.log1p(pd.to_numeric(out["bytes_received"], errors="coerce").clip(lower=0))
    out["is_internal_traffic"] = out["is_internal_traffic"].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0})
    for col in ["src_port", "dst_port"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_pipeline(csv_path: str):
    raw = pd.read_csv(csv_path)
    df = prepare_features(raw)
    X = df[FEATURES]
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
    cv_summary = {
        "pr_auc": average_precision_score(y_train, oof),
        "roc_auc": roc_auc_score(y_train, oof),
    }
    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    results = {
        "test_pr_auc": average_precision_score(y_test, test_proba),
        "test_roc_auc": roc_auc_score(y_test, test_proba),
        "threshold_0_50": evaluate_at_threshold(y_test, test_proba, 0.50),
        "threshold_f2": evaluate_at_threshold(y_test, test_proba, F2_THRESHOLD),
        "cv_summary": cv_summary,
        "classification_report_0_50": classification_report(y_test, (test_proba >= 0.50).astype(int), zero_division=0),
        "classification_report_f2": classification_report(y_test, (test_proba >= F2_THRESHOLD).astype(int), zero_division=0),
    }
    return pipe, results, (X_train, X_test, y_train, y_test, test_proba), df
