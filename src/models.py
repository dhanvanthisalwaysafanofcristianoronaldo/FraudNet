from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS


def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df["is_fraud"].astype(int)
    # Use class weighting to handle imbalance.
    positives = max(1, int(y.sum()))
    negatives = max(1, int((y == 0).sum()))
    scale_pos_weight = negatives / positives

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            scale_pos_weight=scale_pos_weight,
        )),
    ])
    clf.fit(X, y)

    anomaly = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=180, contamination=min(0.12, max(0.02, positives/len(y))), random_state=42)),
    ])
    anomaly.fit(X)
    return clf, anomaly


def score_models(df: pd.DataFrame, clf, anomaly) -> pd.DataFrame:
    out = df.copy()
    X = out[FEATURE_COLUMNS]
    out["ml_score"] = clf.predict_proba(X)[:, 1] * 100
    raw = -anomaly.decision_function(X)
    lo, hi = np.percentile(raw, [2, 98])
    out["anomaly_score"] = np.clip((raw - lo) / max(1e-9, hi - lo) * 100, 0, 100)
    return out
