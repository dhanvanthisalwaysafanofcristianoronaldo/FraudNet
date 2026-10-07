from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").copy()
    # Similar transactions arriving close together at the same merchant/beneficiary are suspicious.
    key = out["merchant_id"].astype(str) + "|" + out["beneficiary_id"].astype(str)
    out["pattern_key"] = key
    counts = out.groupby("pattern_key")["transaction_id"].transform("count").clip(1, 25)
    out["temporal_score"] = np.clip((counts - 1) / 8 * 70, 0, 70)

    # Burst score by account in a rolling 5-minute approximation using shifted timestamps.
    out["prev_ts"] = out.groupby("account_id")["timestamp"].shift(1)
    delta = (out["timestamp"] - out["prev_ts"]).dt.total_seconds().fillna(999999)
    out["burst_score"] = np.where(delta <= 300, np.clip((300 - delta) / 300 * 30, 0, 30), 0)
    out["temporal_score"] = np.clip(out["temporal_score"] + out["burst_score"], 0, 100)
    return out.drop(columns=["pattern_key", "prev_ts", "burst_score"], errors="ignore")


def fuse_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = np.clip(
        0.40 * out["ml_score"]
        + 0.20 * out["anomaly_score"]
        + 0.25 * out["network_score"]
        + 0.15 * out["temporal_score"],
        0, 100,
    )
    out["risk_score"] = out["risk_score"].round(1)
    out["risk_level"] = pd.cut(
        out["risk_score"], bins=[-1, 30, 60, 80, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).astype(str)
    return out


def account_scores(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("account_id").agg(
        transaction_risk=("risk_score", "mean"),
        max_transaction_risk=("risk_score", "max"),
        transaction_count=("transaction_id", "count"),
        network_score=("network_score", "max"),
        anomaly_score=("anomaly_score", "mean"),
        temporal_score=("temporal_score", "mean"),
        total_amount=("amount", "sum"),
    ).reset_index()
    agg["account_risk"] = np.clip(
        0.45 * agg["transaction_risk"]
        + 0.30 * agg["network_score"]
        + 0.15 * agg["anomaly_score"]
        + 0.10 * agg["temporal_score"], 0, 100
    ).round(1)
    return agg.sort_values("account_risk", ascending=False)


def evidence(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("amount_deviation", 0) >= 2.5:
        reasons.append(f"Amount is {row['amount_deviation']:.1f}× the account baseline")
    if row.get("device_account_count", 1) >= 3:
        reasons.append(f"Device is shared across {int(row['device_account_count'])} accounts")
    if row.get("beneficiary_account_count", 1) >= 3:
        reasons.append(f"Beneficiary is connected to {int(row['beneficiary_account_count'])} accounts")
    if row.get("merchant_account_count", 1) >= 5:
        reasons.append(f"Merchant is connected to {int(row['merchant_account_count'])} accounts")
    if row.get("velocity_1h", 1) >= 5:
        reasons.append(f"High account velocity: {int(row['velocity_1h'])} transactions in a short window")
    if row.get("temporal_score", 0) >= 60:
        reasons.append("Strong temporal coordination with similar transactions")
    if row.get("anomaly_score", 0) >= 70:
        reasons.append("Behavior is an outlier versus the learned population")
    if row.get("network_score", 0) >= 70:
        reasons.append("Strong suspicious network connectivity")
    if not reasons:
        reasons.append("No strong fraud indicators detected")
    return reasons[:6]


def recommended_action(score: float) -> str:
    if score >= 91:
        return "TEMPORARY HOLD + MANUAL INVESTIGATION"
    if score >= 81:
        return "STEP-UP AUTHENTICATION + REVIEW"
    if score >= 61:
        return "ENHANCED MONITORING"
    if score >= 31:
        return "MONITOR"
    return "ALLOW"
