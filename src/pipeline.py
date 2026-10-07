from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score

from .features import build_features, FEATURE_COLUMNS
from .models import train_models, score_models
from .graph_engine import account_network_risk, build_graph, detect_fraud_rings
from .risk_engine import temporal_scores, fuse_risk, account_scores, evidence, recommended_action


def run_pipeline(raw: pd.DataFrame):
    df = build_features(raw)
    # Train on the same generated dataset for a hackathon prototype; the graph/temporal layers add independent evidence.
    clf, anomaly = train_models(df)
    scored = score_models(df, clf, anomaly)
    scored = account_network_risk(scored)
    scored = temporal_scores(scored)
    scored = fuse_risk(scored)
    scored["action"] = scored["risk_score"].apply(recommended_action)
    scored["evidence"] = scored.apply(evidence, axis=1)

    accounts = account_scores(scored)
    rings = detect_fraud_rings(scored)
    G = build_graph(scored)

    y = scored["is_fraud"].astype(int)
    p = (scored["risk_score"] >= 60).astype(int)
    metrics = {
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score(y, scored["risk_score"] / 100) if y.nunique() > 1 else 0.0,
        "pr_auc": average_precision_score(y, scored["risk_score"] / 100) if y.nunique() > 1 else 0.0,
    }
    return scored, accounts, rings, G, metrics, clf
