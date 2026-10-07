from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(seed: int = 42, n_transactions: int = 6000) -> pd.DataFrame:
    """Generate reproducible financial transactions with coordinated fraud rings."""
    rng = np.random.default_rng(seed)
    n_accounts = 900
    n_devices = 180
    n_merchants = 250
    n_beneficiaries = 350
    n_rings = 18
    ring_size = 10

    accounts = np.array([f"A{1000+i}" for i in range(n_accounts)])
    devices = np.array([f"D{100+i}" for i in range(n_devices)])
    merchants = np.array([f"M{100+i}" for i in range(n_merchants)])
    beneficiaries = np.array([f"B{1000+i}" for i in range(n_beneficiaries)])
    locations = np.array(["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Kolkata", "Coimbatore"])
    categories = np.array(["Retail", "Electronics", "Travel", "Gaming", "Grocery", "Fashion", "Digital Services"])

    start = pd.Timestamp("2026-09-01")
    end = pd.Timestamp("2026-10-07")
    total_seconds = int((end - start).total_seconds())

    account_baseline = rng.lognormal(mean=np.log(700), sigma=0.55, size=n_accounts)
    account_age_days = rng.integers(30, 2500, size=n_accounts)

    # Seed coordinated rings.
    ring_accounts = []
    ring_device = {}
    ring_merchant = {}
    ring_beneficiary = {}
    ring_by_account = {}
    ring_pool = rng.permutation(accounts)[: n_rings * ring_size]
    for r in range(n_rings):
        chosen = ring_pool[r * ring_size:(r + 1) * ring_size]
        ring_accounts.append(chosen)
        ring_device[r] = devices[(r * 7) % n_devices]
        ring_merchant[r] = merchants[(r * 11) % n_merchants]
        ring_beneficiary[r] = beneficiaries[(r * 13) % n_beneficiaries]
        for a in chosen:
            ring_by_account[a] = r

    rows = []
    ring_event_count = min(n_transactions, n_rings * ring_size)
    normal_count = n_transactions - ring_event_count

    # Normal activity.
    for i in range(normal_count):
        ai = int(rng.integers(0, n_accounts))
        account = accounts[ai]
        baseline = account_baseline[ai]
        amount = max(10, rng.lognormal(np.log(baseline), 0.55))
        ts = start + pd.to_timedelta(int(rng.integers(0, total_seconds)), unit="s")
        hour = ts.hour
        device = devices[int(rng.integers(0, n_devices))]
        merchant = merchants[int(rng.integers(0, n_merchants))]
        beneficiary = beneficiaries[int(rng.integers(0, n_beneficiaries))]
        location = locations[int(rng.integers(0, len(locations)))]
        rows.append({
            "transaction_id": f"TXN{i+1:06d}",
            "account_id": account,
            "timestamp": ts,
            "amount": round(float(amount), 2),
            "merchant_id": merchant,
            "device_id": device,
            "location": location,
            "beneficiary_id": beneficiary,
            "account_age_days": int(account_age_days[ai]),
            "merchant_category": categories[int(rng.integers(0, len(categories)))],
            "is_fraud": 0,
            "ring_id": "",
        })

    # Coordinated ring bursts. Individual amounts remain plausible; relationships carry the signal.
    next_id = normal_count + 1
    for r, members in enumerate(ring_accounts, start=0):
        burst_start = start + pd.to_timedelta(int(rng.integers(0, total_seconds - 3600)), unit="s")
        for j in range(ring_size):
            account = members[j]
            ai = int(np.where(accounts == account)[0][0])
            ts = burst_start + pd.to_timedelta(int(j * rng.integers(20, 45)), unit="s")
            amount = max(50, rng.lognormal(np.log(account_baseline[ai] * rng.uniform(1.1, 2.5)), 0.25))
            # Keep the transaction plausible, while making the ring obvious through shared context.
            rows.append({
                "transaction_id": f"TXN{next_id:06d}",
                "account_id": account,
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "merchant_id": ring_merchant[r],
                "device_id": ring_device[r],
                "location": locations[(r + j) % len(locations)],
                "beneficiary_id": ring_beneficiary[r],
                "account_age_days": int(account_age_days[ai]),
                "merchant_category": "Digital Services",
                "is_fraud": 1,
                "ring_id": f"RING-{r+1:02d}",
            })
            next_id += 1

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # A few non-ring hard negatives: unusual but legitimate.
    hard_idx = rng.choice(df.index[df.is_fraud == 0], size=min(120, len(df)//30), replace=False)
    df.loc[hard_idx, "amount"] *= rng.uniform(2.5, 4.5, size=len(hard_idx))
    df.loc[hard_idx, "amount"] = df.loc[hard_idx, "amount"].round(2)
    return df
