# FraudNet — Real-Time Financial Fraud Intelligence

FraudNet is a hackathon-ready Streamlit prototype for HNX26PSI04. It combines:
- transaction-level supervised fraud scoring
- personalized behavioral anomaly detection
- graph-based fraud-ring discovery
- temporal/velocity signals
- explainable evidence and recommended actions
- interactive investigation dashboard

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app generates a reproducible synthetic dataset containing both legitimate activity and coordinated fraud rings, so it works without an external dataset.

## Optional real CSV
Use the sidebar upload. Required columns:
`transaction_id, account_id, timestamp, amount, merchant_id, device_id, location, beneficiary_id, is_fraud`

Optional columns are preserved when present.

## Demo flow
1. Open Overview and show live risk metrics.
2. Open Fraud Rings and select a detected ring.
3. Show the graph and evidence.
4. Open Transactions and inspect a high-risk transaction.
5. Open Explainability and show feature contributions/evidence.
6. Explain that individually normal transactions can become suspicious when network and temporal context are considered.
