from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df.iterrows():
        a = str(r["account_id"])
        d = str(r["device_id"])
        m = str(r["merchant_id"])
        b = str(r["beneficiary_id"])
        G.add_node(a, kind="account")
        for node, kind in [(d, "device"), (m, "merchant"), (b, "beneficiary")]:
            G.add_node(node, kind=kind)
            if G.has_edge(a, node):
                G[a][node]["weight"] += 1
            else:
                G.add_edge(a, node, weight=1)

    # Also add account-account edges for shared entities so an investigator can
    # directly see the relationship network when filtering to account nodes.
    for col, weight in [("device_id", 4.0), ("merchant_id", 2.0), ("beneficiary_id", 5.0)]:
        for _, grp in df.groupby(col):
            members = grp["account_id"].astype(str).unique().tolist()
            if len(members) > 30:
                continue
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += weight
                    else:
                        G.add_edge(a, b, weight=weight, relation=col)
    return G


def account_network_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    device_shared = out.groupby("device_id")["account_id"].transform("nunique").clip(1, 20)
    merchant_shared = out.groupby("merchant_id")["account_id"].transform("nunique").clip(1, 30)
    beneficiary_shared = out.groupby("beneficiary_id")["account_id"].transform("nunique").clip(1, 30)
    network = (
        np.minimum(device_shared / 8, 1) * 35
        + np.minimum(merchant_shared / 12, 1) * 20
        + np.minimum(beneficiary_shared / 10, 1) * 45
    )
    out["network_score"] = np.clip(network, 0, 100)
    return out


def detect_fraud_rings(df: pd.DataFrame, min_accounts: int = 4) -> pd.DataFrame:
    """Detect candidate rings from repeated multi-entity combinations.

    A candidate requires the same pair of relationship dimensions (for example
    device+beneficiary) to be shared by multiple accounts. This is more selective
    than linking accounts on a common merchant alone.
    """
    candidate_sets = {}
    for keys in [("device_id", "beneficiary_id"), ("device_id", "merchant_id"), ("merchant_id", "beneficiary_id")]:
        for combo, grp in df.groupby(list(keys)):
            accounts = sorted(grp["account_id"].astype(str).unique())
            if min_accounts <= len(accounts) <= 30:
                candidate_sets[tuple(accounts)] = {"keys": keys, "combo": combo}

    rows = []
    for idx, (accounts, meta) in enumerate(candidate_sets.items(), start=1):
        member_tx = df[df["account_id"].astype(str).isin(accounts)]
        keys = meta["keys"]
        shared_counts = {k: member_tx[k].nunique() for k in ["device_id", "merchant_id", "beneficiary_id"]}
        sharedness = sum(1 for k in keys if member_tx[k].nunique() == 1)
        temporal = float(member_tx["temporal_score"].mean()) if "temporal_score" in member_tx else 0.0
        score = float(np.clip(55 + len(accounts) * 3 + sharedness * 10 + temporal * 0.2, 0, 100))
        rows.append({
            "ring_id": f"DISC-{idx:02d}",
            "account_count": len(accounts),
            "accounts": accounts,
            "network_strength": round(score, 1),
            "evidence_keys": ", ".join(keys),
            "shared_counts": shared_counts,
        })
    return pd.DataFrame(rows).sort_values("network_strength", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["ring_id","account_count","accounts","network_strength","evidence_keys","shared_counts"])
