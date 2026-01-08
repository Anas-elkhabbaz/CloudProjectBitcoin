import os
import io
import pandas as pd
import streamlit as st

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

st.set_page_config(page_title="BTC Signals & KPIs", layout="wide")

# --------- CONFIG ----------
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT", "stfinancedl13131")
CONTAINER = "datalake"
PARQUET_PREFIX = "btc/predictions/predictions_5m_nodv/"

PROBA_BUY = float(os.getenv("PROBA_BUY", "0.60"))
PROBA_SELL = float(os.getenv("PROBA_SELL", "0.40"))
N_ROWS = int(os.getenv("N_ROWS", "2000"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "0"))

# --------- AUTH (Managed Identity) ----------
@st.cache_resource
def get_credential():
    return DefaultAzureCredential()

def signal_from_proba(p: float) -> str:
    if p >= PROBA_BUY:
        return "🟢 BUY"
    if p <= PROBA_SELL:
        return "🔴 SELL"
    return "🟡 HOLD"

# --------- DIRECT PARQUET READ (bypassing Delta) ----------
@st.cache_data(ttl=30)
def load_predictions_batched() -> pd.DataFrame:
    """
    Read parquet files directly (bypassing Delta due to path mismatch in Delta log)
    """
    cred = get_credential()
    
    blob_service = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=cred
    )
    
    container = blob_service.get_container_client(CONTAINER)
    
    # Get all parquet files
    blobs = list(container.list_blobs(name_starts_with=PARQUET_PREFIX))
    parquet_files = [b for b in blobs if b.name.endswith('.parquet') and '_delta_log' not in b.name]
    
    if not parquet_files:
        raise ValueError(f"No parquet files found at {PARQUET_PREFIX}")
    
    # Sort by last modified, get recent files
    parquet_files.sort(key=lambda x: x.last_modified, reverse=True)
    
    # Read files (limit to avoid memory issues)
    dfs = []
    max_files = 50 if LOOKBACK_HOURS == 0 else 100  # Read more files if filtering by time
    
    for blob in parquet_files[:max_files]:
        try:
            blob_client = container.get_blob_client(blob.name)
            stream = blob_client.download_blob()
            df_chunk = pd.read_parquet(io.BytesIO(stream.readall()))
            dfs.append(df_chunk)
        except Exception as e:
            # Skip problematic files
            continue
    
    if not dfs:
        raise ValueError("Could not read any parquet files")
    
    pdf = pd.concat(dfs, ignore_index=True)
    
    # Process timestamps
    for c in ["event_time_ts", "scoring_time"]:
        if c in pdf.columns:
            pdf[c] = pd.to_datetime(pdf[c], errors="coerce", utc=True)
    
    pdf = pdf.dropna(subset=["event_time_ts"])
    
    # Apply lookback filter if set
    if LOOKBACK_HOURS > 0:
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=LOOKBACK_HOURS)
        pdf = pdf[pdf["event_time_ts"] >= cutoff]
    
    # Keep only latest N rows
    pdf = pdf.sort_values("event_time_ts", ascending=False).head(N_ROWS)
    
    return pdf.reset_index(drop=True)

# --------- UI ----------
st.title("📈 BTCUSDT – Signals & KPIs (from Parquet files)")

st.caption(f"Reading from: {STORAGE_ACCOUNT}/{PARQUET_PREFIX}")

# Load button (prevents crash loops and lets you see real errors)
if "loaded" not in st.session_state:
    st.session_state.loaded = False

colA, colB = st.columns([1, 2])
with colA:
    if st.button("Load predictions"):
        st.session_state.loaded = True

with colB:
    st.info(
        "Reading parquet files directly. Set LOOKBACK_HOURS (e.g., 24 or 48) in Azure App Settings "
        "to reduce data loaded."
    )

if not st.session_state.loaded:
    st.stop()

with st.spinner("Reading parquet files..."):
    try:
        pdf = load_predictions_batched()
    except Exception as e:
        st.error("App started, but reading parquet files failed.")
        st.exception(e)
        st.markdown(
            """
**Common fixes**
- Enable Web App **Identity** (System assigned = ON)
- Give it **Storage Blob Data Reader** on the Storage Account / Container
- Verify `STORAGE_ACCOUNT` is correct
- Check that parquet files exist at the path
            """
        )
        st.stop()

if pdf.empty:
    st.warning("No predictions returned. Check the parquet files and (optional) LOOKBACK_HOURS.")
    st.stop()

latest = pdf.iloc[0]
latest_proba = float(latest["proba_up"]) if "proba_up" in pdf.columns else None
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
        chart_df = (
            pdf.sort_values("event_time_ts")[["event_time_ts", "proba_up"]]
            .set_index("event_time_ts")
        )
        st.line_chart(chart_df)

with right:
    st.subheader("KPIs rapides")
    if "proba_up" in pdf.columns:
        sig = pdf["proba_up"].apply(signal_from_proba)
        st.write(sig.value_counts())

    if "model_run_id" in pdf.columns:
        mr = str(latest.get("model_run_id", ""))
        st.write("Model run id:", (mr[:24] + "...") if len(mr) > 24 else mr)

st.divider()
st.subheader("Dernières lignes")
st.dataframe(pdf.head(50), use_container_width=True)
