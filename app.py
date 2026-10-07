from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_synthetic_data
from src.pipeline import run_pipeline
from src.risk_engine import recommended_action

st.set_page_config(page_title="FraudNet | Financial Fraud Intelligence", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {border:1px solid #e6e6e6;border-radius:14px;padding:16px;background:#ffffff;}
.small {font-size:0.88rem;color:#666;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_data(seed: int, n: int):
    return generate_synthetic_data(seed=seed, n_transactions=n)

@st.cache_resource(show_spinner=False)
def cached_pipeline(df_hash: int, raw: pd.DataFrame):
    return run_pipeline(raw)

st.sidebar.title("🛡️ FraudNet")
st.sidebar.caption("Real-Time Financial Fraud Intelligence")
source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"])
seed = st.sidebar.number_input("Synthetic seed", 1, 9999, 42)
n = st.sidebar.slider("Synthetic transactions", 2000, 10000, 6000, 500)

if source == "Synthetic demo":
    raw = cached_data(int(seed), int(n))
else:
    uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with: transaction_id, account_id, timestamp, amount, merchant_id, device_id, location, beneficiary_id, is_fraud")
        st.stop()
    raw = pd.read_csv(uploaded)

with st.spinner("Building fraud intelligence graph and risk models..."):
    df_hash = int(pd.util.hash_pandas_object(raw.astype(str), index=True).sum())
    scored, accounts, rings, graph, metrics, clf = cached_pipeline(df_hash, raw)

st.title("🛡️ FraudNet")
st.markdown("**From transaction-level fraud detection to coordinated fraud-ring intelligence.**")

pages = ["Overview", "Live Transactions", "Fraud Rings", "Account Investigation", "Explainability"]
page = st.sidebar.radio("Navigate", pages)

if page == "Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", f"{len(scored):,}")
    c2.metric("High/Critical", f"{int((scored.risk_score >= 61).sum()):,}")
    c3.metric("Fraud Rings", f"{len(rings):,}")
    c4.metric("Amount at Risk", f"₹{scored.loc[scored.risk_score >= 61, 'amount'].sum()/100000:.1f}L")
    c5.metric("Critical", f"{int((scored.risk_score >= 81).sum()):,}")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Risk distribution")
        risk_counts = scored["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0).reset_index()
        risk_counts.columns = ["Risk", "Transactions"]
        fig = px.bar(risk_counts, x="Risk", y="Transactions", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Model quality")
        metric_df = pd.DataFrame({"Metric": ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], "Score": [metrics[k] for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]]})
        fig = px.bar(metric_df, x="Metric", y="Score", range_y=[0,1], text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why FraudNet is different")
    st.info("A transaction can look normal in isolation. FraudNet combines supervised ML, behavioral anomaly detection, graph relationships and temporal coordination to identify coordinated patterns, then explains the evidence and recommends an action.")

elif page == "Live Transactions":
    st.subheader("🔴 Live Transaction Monitor")
    st.caption("Synthetic stream for demonstration. In production, replace this input with Kafka/webhook/event-stream ingestion.")
    filters = st.multiselect("Risk levels", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], default=["HIGH", "CRITICAL"])
    view = scored[scored.risk_level.isin(filters)].sort_values("timestamp", ascending=False).copy()
    cols = ["transaction_id", "account_id", "timestamp", "amount", "risk_score", "risk_level", "action", "merchant_id", "device_id", "beneficiary_id"]
    st.dataframe(view[cols].head(250), use_container_width=True, hide_index=True)

    st.subheader("Selected transaction")
    if len(view):
        tid = st.selectbox("Transaction ID", view.transaction_id.head(100).tolist())
        row = view[view.transaction_id == tid].iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("Risk", f"{row.risk_score:.1f}/100")
        b.metric("ML", f"{row.ml_score:.1f}")
        c.metric("Network", f"{row.network_score:.1f}")
        d.metric("Temporal", f"{row.temporal_score:.1f}")
        st.warning(f"**Action:** {row.action}") if row.risk_score >= 61 else st.success(f"**Action:** {row.action}")
        st.write("**Evidence:**")
        for reason in row.evidence:
            st.write(f"• {reason}")

elif page == "Fraud Rings":
    st.subheader("🕸️ Fraud Ring Intelligence")
    st.caption("Clusters are built from shared devices, merchants and beneficiaries. The demo data contains seeded coordinated rings.")
    if rings.empty:
        st.success("No multi-account rings detected.")
        st.stop()
    ring_display = rings[["ring_id", "account_count", "network_strength"]].copy().sort_values("network_strength", ascending=False)
    st.dataframe(ring_display, use_container_width=True, hide_index=True)
    selected = st.selectbox("Investigate ring", ring_display.ring_id.tolist())
    ring = rings[rings.ring_id == selected].iloc[0]
    members = ring["accounts"]
    ring_tx = scored[scored.account_id.isin(members)].copy()
    ring_risk = float(ring_tx.risk_score.mean()) if len(ring_tx) else float(ring.network_strength)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ring Risk", f"{ring_risk:.1f}/100")
    c2.metric("Members", len(members))
    c3.metric("Transactions", len(ring_tx))
    c4.metric("Amount", f"₹{ring_tx.amount.sum():,.0f}")

    st.markdown("### Fraud DNA")
    def shared_entity_score(series, member_count):
        top = int(series.value_counts().iloc[0]) if len(series) else 0
        return min(100.0, 100.0 * top / max(1, member_count))

    dna = {
        "Shared device": shared_entity_score(ring_tx.device_id, len(members)),
        "Shared merchant": shared_entity_score(ring_tx.merchant_id, len(members)),
        "Shared beneficiary": shared_entity_score(ring_tx.beneficiary_id, len(members)),
        "Temporal coordination": float(ring_tx.temporal_score.mean()) if len(ring_tx) else 0,
        "Network strength": float(ring.network_strength),
    }
    dna_df = pd.DataFrame({"Signal": list(dna), "Score": list(dna.values())})
    st.plotly_chart(px.bar(dna_df, x="Signal", y="Score", range_y=[0,100], text_auto=".0f"), use_container_width=True)

    # Build an account-only graph for the selected ring.
    sub = graph.subgraph(members).copy()
    edge_x, edge_y = [], []
    pos = __import__("networkx").spring_layout(sub, seed=42)
    for u,v in sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line=dict(width=1))
    node_x = [pos[n][0] for n in sub.nodes()]
    node_y = [pos[n][1] for n in sub.nodes()]
    node_risk = [float(accounts.loc[accounts.account_id == n, "account_risk"].iloc[0]) if n in set(accounts.account_id) else 50 for n in sub.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(sub.nodes()), textposition="top center", marker=dict(size=18, color=node_risk, colorscale="Turbo", showscale=True, colorbar=dict(title="Risk")), hovertext=[f"Account {n}" for n in sub.nodes()])
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(height=560, title="Account relationship graph", xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ring evidence")
    for item in [
        f"{len(members)} accounts form a connected suspicious cluster.",
        f"Average transaction risk inside cluster: {ring_tx.risk_score.mean():.1f}/100.",
        f"Shared devices: {ring_tx.device_id.nunique()} | merchants: {ring_tx.merchant_id.nunique()} | beneficiaries: {ring_tx.beneficiary_id.nunique()}.",
        "Multiple transactions occur within a compressed time window, increasing coordination risk.",
    ]:
        st.write(f"• {item}")

elif page == "Account Investigation":
    st.subheader("🔎 Account Investigation")
    account_options = accounts.account_id.head(150).tolist()
    account_id = st.selectbox("Account", account_options)
    ar = accounts[accounts.account_id == account_id].iloc[0]
    tx = scored[scored.account_id == account_id].sort_values("timestamp")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Account Risk", f"{ar.account_risk:.1f}/100")
    c2.metric("Transactions", int(ar.transaction_count))
    c3.metric("Total Amount", f"₹{ar.total_amount:,.0f}")
    c4.metric("Network Risk", f"{ar.network_score:.1f}")
    st.write("**Recommended action:**", recommended_action(float(ar.account_risk)))
    st.markdown("### Investigation timeline")
    st.dataframe(tx[["timestamp","transaction_id","amount","merchant_id","device_id","location","beneficiary_id","risk_score","risk_level","action"]], use_container_width=True, hide_index=True)

elif page == "Explainability":
    st.subheader("🧠 Explainable AI")
    st.caption("Evidence is generated from model signals and relationship/behavioral rules. This keeps every score auditable for the evaluator.")
    high = scored.sort_values("risk_score", ascending=False)
    tid = st.selectbox("Transaction", high.transaction_id.head(200).tolist())
    row = high[high.transaction_id == tid].iloc[0]
    st.metric("Final risk", f"{row.risk_score:.1f}/100")
    st.markdown("### Risk composition")
    components = pd.DataFrame({
        "Signal": ["ML fraud probability", "Behavior anomaly", "Network risk", "Temporal coordination"],
        "Score": [row.ml_score, row.anomaly_score, row.network_score, row.temporal_score],
        "Weight": [0.40, 0.20, 0.25, 0.15],
    })
    components["Weighted contribution"] = components.Score * components.Weight
    st.dataframe(components.round(2), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(components, x="Signal", y="Weighted contribution", text_auto=".1f"), use_container_width=True)
    st.markdown("### Evidence")
    for reason in row.evidence:
        st.write(f"• {reason}")
    st.markdown("### Counterfactual intuition")
    if row.amount_deviation > 1:
        counter = min(row.risk_score, row.risk_score - min(25, (row.amount_deviation - 1) * 8))
        st.info(f"If the transaction amount moved closer to the account's normal baseline while other signals stayed constant, the estimated risk could fall toward **{max(0, counter):.0f}/100**. This is an illustrative sensitivity explanation, not a causal guarantee.")
    else:
        st.info("This transaction is already close to the account's normal amount baseline; network and temporal evidence are more important here.")

st.sidebar.divider()
st.sidebar.caption("Hackathon prototype • Fraud ring intelligence • Explainable risk")
