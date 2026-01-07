import os
import pandas as pd
import streamlit as st

from azure.identity import DefaultAzureCredential
from deltalake import DeltaTable

st.set_page_config(page_title="BTC Signals & KPIs", layout="wide")

# --------- CONFIG ----------
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT", "stfinancedl13131")
CONTAINER = os.getenv("CONTAINER", "datalake")

# chemin Delta (format delta-rs azure): az://<container>/<path>
PRED_DELTA_PATH = os.getenv(
    "PRED_DELTA_PATH",
    "az://datalake/btc/predictions/predictions_5m"
)

PROBA_BUY = float(os.getenv("PROBA_BUY", "0.60"))
PROBA_SELL = float(os.getenv("PROBA_SELL", "0.40"))
N_ROWS = int(os.getenv("N_ROWS", "2000"))  # charge les derniers N en triant ensuite

# --------- AUTH (Managed Identity) ----------
@st.cache_resource
def get_credential():
    # Sur App Service, DefaultAzureCredential utilise la Managed Identity automatiquement
    return DefaultAzureCredential()

def get_storage_token() -> str:
    cred = get_credential()
    # scope standard pour Azure Storage
    return cred.get_token("https://storage.azure.com/.default").token

# --------- READ DELTA ----------
@st.cache_data(ttl=30)
def load_predictions() -> pd.DataFrame:
    token = get_storage_token()
    dt = DeltaTable(
        PRED_DELTA_PATH,
        storage_options={
            "account_name": STORAGE_ACCOUNT,
            "token": token,  # token AAD
        }
    )
    pdf = dt.to_pandas()  # si ça devient gros, on optimisera (filters/partition)
    if pdf.empty:
        return pdf

    # normaliser timestamps
    for c in ["event_time_ts", "scoring_time"]:
        if c in pdf.columns:
            pdf[c] = pd.to_datetime(pdf[c], errors="coerce", utc=True)

    # garder les colonnes utiles si présentes
    cols = [c for c in ["symbol","interval","event_time_ts","proba_up","pred_up","model_run_id","scoring_time"] if c in pdf.columns]
    pdf = pdf[cols].dropna(subset=["event_time_ts"]).sort_values("event_time_ts", ascending=False)

    # limiter
    return pdf.head(N_ROWS)

def signal_from_proba(p: float) -> str:
    if p >= PROBA_BUY:
        return "🟢 BUY"
    if p <= PROBA_SELL:
        return "🔴 SELL"
    return "🟡 HOLD"

# --------- UI ----------
st.title("📈 BTCUSDT – Signals & KPIs (from Delta Lake)")

pdf = load_predictions()

if pdf.empty:
    st.warning("Aucune prédiction trouvée dans la table Delta. Vérifie que le job d'inférence tourne et que le chemin est correct.")
    st.stop()

latest = pdf.iloc[0]
latest_proba = float(latest["proba_up"]) if "proba_up" in latest else None
latest_signal = signal_from_proba(latest_proba) if latest_proba is not None else "N/A"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dernière proba_up", f"{latest_proba:.3f}" if latest_proba is not None else "N/A")
c2.metric("Signal", latest_signal)
c3.metric("Nb prédictions (chargées)", f"{len(pdf)}")
c4.metric("Dernier event_time", str(latest.get("event_time_ts", "")))

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Historique proba_up")
    if "event_time_ts" in pdf.columns and "proba_up" in pdf.columns:
        chart_df = pdf.sort_values("event_time_ts")[["event_time_ts", "proba_up"]].set_index("event_time_ts")
        st.line_chart(chart_df)

with right:
    st.subheader("KPIs rapides")
    # distribution des signaux
    if "proba_up" in pdf.columns:
        sig = pdf["proba_up"].apply(signal_from_proba)
        st.write(sig.value_counts())

    if "model_run_id" in pdf.columns:
        st.write("Model run id:", str(latest.get("model_run_id", ""))[:24] + "...")

st.divider()
st.subheader("Dernières lignes")
st.dataframe(pdf.head(50), use_container_width=True)
st.caption(f"Chemin Delta Lake: {PRED_DELTA_PATH}")