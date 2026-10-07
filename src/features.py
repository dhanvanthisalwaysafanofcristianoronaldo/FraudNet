from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp", "amount", "account_id"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["transaction_id", "merchant_id", "device_id", "location", "beneficiary_id"]:
        if col not in out:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    if "is_fraud" not in out:
        out["is_fraud"] = 0
    out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_transactions(df)
    g = out.groupby("account_id", sort=False)
    out["account_txn_count"] = g["amount"].transform("count")
    out["account_avg_amount"] = g["amount"].transform("mean").clip(lower=1)
    out["account_std_amount"] = g["amount"].transform("std").fillna(1).clip(lower=1)
    out["amount_deviation"] = out["amount"] / out["account_avg_amount"]
    out["amount_z"] = ((out["amount"] - out["account_avg_amount"]) / out["account_std_amount"]).clip(-20, 20)
    out["hour"] = out["timestamp"].dt.hour
    out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 23)).astype(int)
    out["weekday"] = out["timestamp"].dt.weekday

    out["new_device_proxy"] = (out.groupby("account_id")["device_id"].transform("nunique") > 1).astype(int)
    out["device_account_count"] = out.groupby("device_id")["account_id"].transform("nunique")
    out["beneficiary_account_count"] = out.groupby("beneficiary_id")["account_id"].transform("nunique")
    out["merchant_account_count"] = out.groupby("merchant_id")["account_id"].transform("nunique")

    # Robust per-account one-hour velocity without relying on duplicate timestamp indexes.
    velocity = np.ones(len(out), dtype=float)
    for _, idx in out.groupby("account_id", sort=False).groups.items():
        times = out.loc[idx, "timestamp"].sort_values()
        vals = []
        for ts in times:
            vals.append(float(((times >= ts - pd.Timedelta(hours=1)) & (times <= ts)).sum()))
        velocity[times.index.to_numpy()] = vals
    out["velocity_1h"] = np.clip(velocity, 1, 100)

    numeric_cols = [
        "amount", "account_age_days", "account_txn_count", "account_avg_amount",
        "account_std_amount", "amount_deviation", "amount_z", "hour", "is_night",
        "weekday", "new_device_proxy", "device_account_count",
        "beneficiary_account_count", "merchant_account_count", "velocity_1h"
    ]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


FEATURE_COLUMNS = [
    "amount", "account_age_days", "account_txn_count", "account_avg_amount",
    "account_std_amount", "amount_deviation", "amount_z", "hour", "is_night",
    "weekday", "new_device_proxy", "device_account_count",
    "beneficiary_account_count", "merchant_account_count", "velocity_1h"
]
