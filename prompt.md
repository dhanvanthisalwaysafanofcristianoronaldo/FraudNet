# Master Prompt for an AI Coding Agent

You are a senior Python/ML engineer building a hackathon prototype called **FraudNet — Real-Time Financial Fraud Intelligence** for problem statement HNX26PSI04.

## Objective
Build a working web application that detects both individual suspicious transactions and coordinated fraud rings. The key challenge is that a single transaction may look legitimate while a pattern across accounts, devices, merchants, beneficiaries and time reveals fraud.

## Required capabilities
1. Generate or ingest transaction data.
2. Train a supervised fraud classifier (prefer XGBoost) with class-imbalance handling.
3. Add unsupervised anomaly detection (Isolation Forest).
4. Engineer personalized behavioral features: account baseline amount, amount deviation/z-score, velocity, new/shared device, shared merchant, shared beneficiary, account age, time-of-day.
5. Build a relationship graph with NetworkX. Nodes may represent accounts/devices/merchants/beneficiaries. Edges represent usage or transaction relationships.
6. Detect multi-account suspicious clusters/fraud rings.
7. Add temporal coordination detection: bursts, repeated merchant/beneficiary patterns and short time windows.
8. Fuse scores into a 0–100 risk score using a transparent formula. Default weights: ML 40%, anomaly 20%, network 25%, temporal 15%.
9. Produce both transaction risk and account risk.
10. Give an evidence-based explanation for every score.
11. Show recommended action: ALLOW, MONITOR, ENHANCED MONITORING, STEP-UP AUTHENTICATION + REVIEW, or TEMPORARY HOLD + MANUAL INVESTIGATION.
12. Provide an interactive dashboard with Overview, Live Transactions, Fraud Rings, Account Investigation and Explainability pages.
13. Visualize fraud-ring relationships as an interactive graph.
14. Show Fraud DNA: shared device, shared merchant, shared beneficiary, temporal coordination and network strength.
15. Include a controlled synthetic dataset with seeded coordinated fraud rings so the demo works without external data.
16. Include evaluation metrics: precision, recall, F1, ROC-AUC and PR-AUC. Do not optimize for accuracy alone.
17. Clearly label synthetic streaming as a simulation. Do not claim a production banking integration.

## Engineering constraints
- Python 3.11+.
- Streamlit frontend for the hackathon MVP.
- Pandas, NumPy, scikit-learn, XGBoost, NetworkX and Plotly.
- SHAP may be used where practical, but the application must still run if SHAP explanations are unavailable; use transparent evidence/risk-component explanations as fallback.
- Keep modules separated: data generation, feature engineering, models, graph engine, temporal/risk engine and Streamlit UI.
- Make the application reproducible with a fixed random seed.
- Avoid hard-coded fake metrics. Compute all displayed metrics from the data.
- Do not leak labels into features. Fraud labels may only be used for supervised training/evaluation.
- Do not use future information to create real-time features. Clearly mark any demo-only batch features.
- Every risk score must have an explanation.
- The UI must not flag every unusual transaction as fraud; preserve legitimate hard negatives and evaluate precision.

## Demo story
Create a seeded scenario with around 10 accounts that individually have plausible transactions but collectively share infrastructure and/or transaction timing. The demo should show:
- individually moderate transaction risk
- a suspicious shared device/merchant/beneficiary pattern
- compressed transaction timing
- a detected ring with high network risk
- an evidence panel explaining the ring
- a recommended action

## Output
Return a complete runnable project with:
- app.py
- src/ modules
- requirements.txt
- README.md
- sample/schema CSV
- clear run instructions

Before finishing, validate imports, run the application/pipeline on synthetic data, verify there are no NaNs/infinite scores, and verify at least one fraud ring is detected.
