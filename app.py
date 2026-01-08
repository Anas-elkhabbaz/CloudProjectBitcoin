import os
import pandas as pd
import streamlit as st

from azure.identity import DefaultAzureCredential
from deltalake import DeltaTable
import pyarrow.dataset as ds

st.set_page_config(page_title="BTC Signals & KPIs", layout="wide")

# --------- CONFIG ----------
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT", "stfinancedl13131")

# delta-rs azure path: az://<container>/<path>

PRED_DELTA_PATH = os.getenv(
    "PRED_DELTA_PATH",
    "az://datalake/btc/predictions/predictions_5m_nodv"
)


PROBA_BUY = float(os.getenv("PROBA_BUY", "0.60"))
PROBA_SELL = float(os.getenv("PROBA_SELL", "0.40"))
N_ROWS = int(os.getenv("N_ROWS", "2000"))

# Optional: limit scan window (reduces work). Set in Azure App Settings if you want.
# Example: LOOKBACK_HOURS=48
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "0"))

# --------- AUTH (Managed Identity) ----------
@st.cache_resource
def get_credential():
    return DefaultAzureCredential()

def get_storage_token() -> str:
    cred = get_credential()
    return cred.get_token("https://storage.azure.com/.default").token

def signal_from_proba(p: float) -> str:
    if p >= PROBA_BUY:
        return "🟢 BUY"
    if p <= PROBA_SELL:
        return "🔴 SELL"
    return "🟡 HOLD"

# --------- MEMORY-SAFE DELTA READ ----------
@st.cache_data(ttl=30)
def load_predictions_batched() -> pd.DataFrame:
    """
    Read Delta table in batches, keep only the latest N rows by event_time_ts.
    This avoids loading the full table into memory (dt.to_pandas()).
    """
    token = get_storage_token()

    dt = DeltaTable(
        PRED_DELTA_PATH,
        storage_options={
            "account_name": STORAGE_ACCOUNT,
            "token": token,
        }
    )

    dataset = dt.to_pyarrow_dataset()

    # Columns we want (only keep those that exist)
    wanted = ["symbol", "interval", "event_time_ts", "proba_up", "pred_up", "model_run_id", "scoring_time"]
    existing = [c for c in wanted if c in dataset.schema.names]

    if "event_time_ts" not in existing:
        # If your table uses a different timestamp column, change it here.
        raise ValueError(
            "Column 'event_time_ts' not found in Delta table. "
            f"Available columns: {dataset.schema.names}"
        )

    # Optional filter: only last X hours (works if event_time_ts is a timestamp column)
    arrow_filter = None
    if LOOKBACK_HOURS > 0:
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=LOOKBACK_HOURS)
        try:
            arrow_filter = ds.field("event_time_ts") >= cutoff
        except Exception:
            arrow_filter = None  # if type mismatch, skip filter

    # Scan in batches (bounded memory)
    scanner = dataset.scanner(columns=existing, filter=arrow_filter)

    keep = pd.DataFrame(columns=existing)
    for rb in scanner.to_batches():
        chunk = rb.to_pandas()

        # Normalize timestamps safely
        for c in ["event_time_ts", "scoring_time"]:
            if c in chunk.columns:
                chunk[c] = pd.to_datetime(chunk[c], errors="coerce", utc=True)

        chunk = chunk.dropna(subset=["event_time_ts"])
        if chunk.empty:
            continue

        # Append then keep only top N by latest event_time_ts
        keep = pd.concat([keep, chunk], ignore_index=True)
        keep = keep.sort_values("event_time_ts", ascending=False).head(N_ROWS)

    return keep.reset_index(drop=True)

# --------- UI ----------
st.title("📈 BTCUSDT – Signals & KPIs (from Delta Lake)")

st.caption(f"Delta path: {PRED_DELTA_PATH}")

# Load button (prevents crash loops and lets you see real errors)
if "loaded" not in st.session_state:
    st.session_state.loaded = False

colA, colB = st.columns([1, 2])
with colA:
    if st.button("Load predictions"):
        st.session_state.loaded = True

with colB:
    st.info(
        "Tip: If it’s still slow, set LOOKBACK_HOURS (e.g., 24 or 48) in Azure App Settings "
        "to reduce scanning."
    )

if not st.session_state.loaded:
    st.stop()

with st.spinner("Reading Delta table..."):
    try:
        pdf = load_predictions_batched()
    except Exception as e:
        st.error("App started, but reading Delta Lake failed.")
        st.exception(e)
        st.markdown(
            """
**Common fixes**
- Enable Web App **Identity** (System assigned = ON)
- Give it **Storage Blob Data Reader** on the Storage Account / Container
- Verify `STORAGE_ACCOUNT` and `PRED_DELTA_PATH`
- If the table is huge, set `LOOKBACK_HOURS=48` (or smaller)
            """
        )
        st.stop()

if pdf.empty:
    st.warning("No predictions returned. Check the Delta path/job and (optional) LOOKBACK_HOURS.")
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

