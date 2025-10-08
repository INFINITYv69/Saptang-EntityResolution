# app.py
"""
Campus Entity Resolution & Security Monitoring — Streamlit prototype
Ready to push to GitHub.

Features / improvements over original:
- Clearer modular structure with typed functions
- Robust handling when datasets / entities are empty
- Export fused activity CSV
- Configurable threshold constant for "missing" alert
- Minor UX improvements: status card, improved charting, safe selectbox
- Caching used for synthetic data and ER fusion
"""

from datetime import datetime, timedelta
from collections import Counter
from difflib import SequenceMatcher
from typing import Tuple, Dict, Any

import random
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Config & Constants ----------
PAGE_TITLE = "Saptang — Campus ER & Security Monitor"
MISSING_HOURS_THRESHOLD = 12  # hours without observation --> alert
FUZZY_NAME_THRESHOLD = 0.6

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")

# ---------- Small CSS ----------
st.markdown(
    """
    <style>
    .header-title { font-size: 2.2rem; font-weight:700; color:#1E40AF; }
    .subheader { font-size:1.1rem; color:#4B5563; margin-bottom:12px; }
    .card { background:#F8FAFC; padding:12px; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.04); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def similar(a: Any, b: Any) -> float:
    """Return normalized similarity [0,1] between two strings using SequenceMatcher."""
    try:
        if pd.isna(a) or pd.isna(b):
            return 0.0
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    except Exception:
        return 0.0


def score_match(row: Dict[str, Any], profile: Dict[str, Any]) -> float:
    """
    Heuristic confidence score [0..1] matching a log row against a profile.
    Weighted fields: card_id (5), device_hash (4), email (3), name similarity (2).
    """
    s = 0.0
    weight_sum = 0.0

    weight_sum += 5
    s += 5 * (1.0 if row.get("card_id") == profile.get("card_id") else 0.0)

    weight_sum += 4
    s += 4 * (1.0 if row.get("device_hash") == profile.get("device_hash") else 0.0)

    weight_sum += 3
    try:
        s += 3 * (1.0 if str(row.get("email", "")).lower() == str(profile.get("email", "")).lower() else 0.0)
    except Exception:
        s += 0.0

    weight_sum += 2
    s += 2 * similar(row.get("name", ""), profile.get("name", ""))

    return float(s / weight_sum) if weight_sum else 0.0


# ---------- Synthetic Data (cached) ----------
@st.cache_data
def generate_synthetic_data(num_entities: int = 12, days: int = 2, seed: int = 42):
    """Create synthetic profiles, swipes, wifi logs, and notes for demo/test purposes."""
    np.random.seed(seed)
    random.seed(seed)
    now = datetime.now()
    ids = [f"E{1000+i}" for i in range(num_entities)]
    names = [f"Person_{i}" for i in range(num_entities)]
    emails = [f"user{i}@campus.edu" for i in range(num_entities)]
    card_ids = [f"CARD-{2000+i}" for i in range(num_entities)]
    device_hashes = [f"DVC-{3000+i}" for i in range(num_entities)]
    profiles = pd.DataFrame({
        "entity_id": ids,
        "name": names,
        "email": emails,
        "card_id": card_ids,
        "device_hash": device_hashes,
        "dept": np.random.choice(["CSE", "ECE", "ME", "Design"], size=num_entities)
    })

    # logs
    start = now - timedelta(days=days)
    locations = ["Main Library", "North Gate", "CS Lab-1", "Cafeteria", "Dorm A", "Auditorium"]
    aps = ["AP-Lib-1F", "AP-Gate", "AP-CS-2F", "AP-Cafe", "AP-DormB"]

    swipe_rows = []
    wifi_rows = []
    for _ in range(num_entities * 30):
        ent = np.random.choice(profiles.index)
        t = start + timedelta(seconds=np.random.randint(0, int((now - start).total_seconds())))
        swipe_rows.append({
            "card_id": profiles.loc[ent, "card_id"],
            "timestamp": t,
            "location": np.random.choice(locations),
            "source": "SwipeLog",
            "type": "CardAccess"
        })

    for _ in range(num_entities * 20):
        ent = np.random.choice(profiles.index)
        t = start + timedelta(seconds=np.random.randint(0, int((now - start).total_seconds())))
        wifi_rows.append({
            "device_hash": profiles.loc[ent, "device_hash"],
            "timestamp": t,
            "access_point": np.random.choice(aps),
            "source": "WiFiLog",
            "type": "AP_Connection"
        })

    # Add orphan swipe (unknown card)
    swipe_rows.append({
        "card_id": "CARD-9999",
        "timestamp": now - timedelta(hours=5),
        "location": "North Gate",
        "source": "SwipeLog",
        "type": "CardAccess"
    })

    # Force an entity to be absent for > 12 hours for demo
    profiles.loc[1, "card_id"] = "CARD-2001"
    # Remove recent swipes of CARD-2001 except an old one
    swipe_rows = [r for r in swipe_rows if r.get("card_id") != "CARD-2001"]
    swipe_rows.append({
        "card_id": "CARD-2001",
        "timestamp": now - timedelta(hours=14, minutes=30),
        "location": "Dorm A",
        "source": "SwipeLog",
        "type": "CardAccess"
    })

    notes = []
    for i in range(8):
        ent = np.random.choice(profiles.index)
        t = start + timedelta(seconds=np.random.randint(0, int((now - start).total_seconds())))
        notes.append({
            "text": f"Helpdesk: request by {profiles.loc[ent, 'name']}",
            "timestamp": t,
            "name": profiles.loc[ent, "name"],
            "email": profiles.loc[ent, "email"],
            "source": "Notes",
            "type": "FreeText"
        })

    swipes = pd.DataFrame(swipe_rows)
    swipes["timestamp"] = pd.to_datetime(swipes["timestamp"])
    wifi = pd.DataFrame(wifi_rows)
    wifi["timestamp"] = pd.to_datetime(wifi["timestamp"])
    notes = pd.DataFrame(notes)
    notes["timestamp"] = pd.to_datetime(notes["timestamp"])

    return profiles, swipes, wifi, notes


# ---------- Entity Resolution & Fusion (cached) ----------
@st.cache_data
def resolve_and_fuse(profiles: pd.DataFrame, swipes: pd.DataFrame, wifi: pd.DataFrame, notes: pd.DataFrame, fuzzy_thresh: float = FUZZY_NAME_THRESHOLD) -> pd.DataFrame:
    """Merge logs and resolve entity ids using deterministic and probabilistic matching."""
    # Merge deterministic matches
    sw = swipes.merge(profiles[["entity_id", "card_id", "name", "email", "device_hash"]], on="card_id", how="left")
    wi = wifi.merge(profiles[["entity_id", "device_hash", "name", "email", "card_id"]], on="device_hash", how="left")
    no = notes.merge(profiles[["entity_id", "name", "email"]], on=["name", "email"], how="left")

    sw_records = sw.rename(columns={"location": "where"}).assign(description=lambda df: "Swiped at " + df["where"].fillna("Unknown"))
    wi_records = wi.rename(columns={"access_point": "where"}).assign(description=lambda df: "WiFi assoc @ " + df["where"].fillna("Unknown"))
    no_records = no.assign(description=lambda df: "Note: " + no["text"].fillna(""))

    # Ensure consistent columns in notes
    for col in ["card_id", "device_hash", "where"]:
        if col not in no_records.columns:
            no_records[col] = pd.NA

    all_logs = pd.concat([
        sw_records[["entity_id", "timestamp", "source", "type", "description", "card_id", "device_hash", "name", "email", "where"]],
        wi_records[["entity_id", "timestamp", "source", "type", "description", "card_id", "device_hash", "name", "email", "where"]],
        no_records[["entity_id", "timestamp", "source", "type", "description", "card_id", "device_hash", "name", "email", "where"]]
    ], ignore_index=True, sort=False)

    prof_lookup = profiles.set_index("entity_id").to_dict(orient="index")

    def attempt_resolve(r):
        # Deterministic by entity_id
        if pd.notna(r.get("entity_id")):
            return r.get("entity_id"), 1.0
        # Deterministic by email
        if pd.notna(r.get("email")):
            m = profiles[profiles["email"].str.lower() == str(r["email"]).lower()]
            if not m.empty:
                return m.iloc[0]["entity_id"], 0.98
        # Fuzzy by name
        best, best_score = None, 0.0
        for eid, p in prof_lookup.items():
            s = similar(r.get("name", ""), p.get("name", ""))
            if s > best_score:
                best_score = s
                best = eid
        if best_score >= fuzzy_thresh:
            return best, 0.5 + 0.5 * best_score
        return pd.NA, 0.0

    resolved_ids = []
    resolved_conf = []
    for _, row in all_logs.iterrows():
        eid, conf = attempt_resolve(row)
        combined = conf
        try:
            if pd.notna(eid):
                combined = max(combined, score_match(row.to_dict(), profiles[profiles["entity_id"] == eid].to_dict(orient="records")[0]))
        except Exception:
            pass
        resolved_ids.append(eid)
        resolved_conf.append(float(round(combined, 3)))

    all_logs["resolved_entity"] = resolved_ids
    all_logs["match_confidence"] = resolved_conf
    fused = all_logs.copy()
    fused["timestamp"] = pd.to_datetime(fused["timestamp"])
    fused = fused.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return fused


# ---------- Predictive Monitoring ----------
def predict_location(entity_id: str, df: pd.DataFrame) -> Tuple[str, float]:
    """Predict the next likely location for an entity using simple pattern frequency (kNN-like)."""
    entity_df = df[(df["resolved_entity"] == entity_id) & df["where"].notna()].sort_values("timestamp", ascending=False)

    if entity_df.shape[0] < 2:
        return "Not enough history to predict.", 0.0

    last_location = entity_df.iloc[0]["where"]
    subsequent_locations = []
    for i in range(1, len(entity_df)):
        # if we find a prior record with the same 'where', log the location that followed it historically
        if entity_df.iloc[i]["where"] == last_location and i > 0:
            subsequent_locations.append(entity_df.iloc[i - 1]["where"])

    if not subsequent_locations:
        return f"Entity usually stops after being at '{last_location}'.", 0.3

    most_common_next = Counter(subsequent_locations).most_common(1)
    if most_common_next:
        next_location, freq = most_common_next[0]
        confidence = freq / len(subsequent_locations)
        return f"Likely heading to **{next_location}** based on historical patterns.", min(0.95, confidence)

    return "No clear future pattern identified.", 0.2


# ---------- Alerting / Anomaly detection ----------
def check_for_missing_alert(entity_id: str, df: pd.DataFrame, threshold_hours: int = MISSING_HOURS_THRESHOLD) -> Tuple[bool, int, datetime]:
    """Return (is_missing, hours_missing, last_seen_datetime)."""
    entity_df = df[df["resolved_entity"] == entity_id]
    now = datetime.now()

    if entity_df.empty:
        last_seen = now - timedelta(hours=999)  # synthetic fallback
    else:
        last_seen = entity_df["timestamp"].max()

    time_since_last = now - last_seen
    if time_since_last > timedelta(hours=threshold_hours):
        hours = int(time_since_last.total_seconds() // 3600)
        return True, hours, last_seen
    return False, 0, last_seen


def get_all_anomalies(df: pd.DataFrame, profiles: pd.DataFrame, threshold_hours: int = MISSING_HOURS_THRESHOLD) -> pd.DataFrame:
    """Return DataFrame of all profiles not seen in `threshold_hours`."""
    anomalies = []
    for entity_id in profiles["entity_id"].unique():
        is_missing, hours_missing, last_seen = check_for_missing_alert(entity_id, df, threshold_hours)
        if is_missing:
            row = profiles[profiles["entity_id"] == entity_id].iloc[0]
            anomalies.append({
                "Entity ID": entity_id,
                "Name": row["name"],
                "Department": row["dept"],
                "Last Seen": last_seen,
                "Missing Duration (hrs)": hours_missing
            })
    if not anomalies:
        return pd.DataFrame(columns=["Entity ID", "Name", "Department", "Last Seen", "Missing Duration (hrs)"])
    return pd.DataFrame(anomalies).sort_values("Missing Duration (hrs)", ascending=False)


# ---------- App UI ----------
def main():
    st.markdown('<div class="header-title">Campus Entity Resolution & Security Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Proactive prototype — hybrid deterministic + probabilistic entity resolution</div>', unsafe_allow_html=True)

    st.sidebar.title("How It Works")
    st.sidebar.markdown(
        """
        1. Merge logs (swipes, wifi, notes) → fuse into unified timeline.
        2. Use deterministic (card/device/email) + fuzzy name matching for ER.
        3. Score each record with a match confidence.
        4. Predict next likely location with a simple historical pattern model.
        5. Trigger alerts if entity not seen for >12 hours.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: You can export the fused timeline for offline analysis.")

    # Load/generate data
    with st.spinner("Generating demo data..."):
        profiles, swipes, wifi, notes = generate_synthetic_data(num_entities=12, days=2)

    fused = resolve_and_fuse(profiles, swipes, wifi, notes)

    # Top-level summary cards
    col1, col2, col3 = st.columns([1.5, 1, 1])
    col1.markdown(f"<div class='card'><b>Profiles:</b> {len(profiles)}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><b>Logs (swipes):</b> {len(swipes)}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><b>Fused records:</b> {len(fused)}</div>", unsafe_allow_html=True)

    st.markdown("---")
    # Entity selector (safe: include only non-null resolved ids)
    entities = [e for e in sorted(fused["resolved_entity"].dropna().unique().tolist())]
    if not entities:
        st.warning("No resolved entities available in the dataset.")
    else:
        selected = st.selectbox("Select an entity to inspect", options=entities, index=0)
        if selected:
            selected_entity_id = selected
            ent_data = fused[fused["resolved_entity"] == selected_entity_id].sort_values("timestamp", ascending=False)

            # get profile info safely
            profile_row = profiles[profiles["entity_id"] == selected_entity_id]
            selected_name = profile_row["name"].iloc[0] if not profile_row.empty else str(selected_entity_id)

            st.subheader(f"Activity timeline — {selected_name} ({selected_entity_id})")

            # Alert status
            is_missing, hours_missing, last_seen = check_for_missing_alert(selected_entity_id, fused)
            if is_missing:
                st.error(f"🚨 SECURITY ALERT — {selected_name} not observed for {hours_missing} hours (threshold: {MISSING_HOURS_THRESHOLD} hrs).")
            else:
                st.success(f"✅ Last observed: {last_seen.strftime('%Y-%m-%d %H:%M:%S')}", icon="✅")

            # Dataframe view (most recent first)
            st.dataframe(ent_data[["timestamp", "source", "description", "match_confidence"]], use_container_width=True)

            # Confidence trend chart (if enough points)
            if ent_data.shape[0] >= 2:
                chart = alt.Chart(ent_data).mark_line(point=True).encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("match_confidence:Q", title="Match Confidence"),
                    tooltip=["timestamp", "source", "match_confidence"]
                ).properties(height=300, title="Match Confidence Over Time")
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Not enough records to plot confidence trend.")

            # Predictive monitoring
            st.markdown("---")
            st.subheader("Predictive monitoring")
            pred_text, pred_conf = predict_location(selected_entity_id, fused)
            st.info(f"🧭 Prediction: {pred_text}  *(Confidence: {pred_conf:.2f})*")

            st.caption(f"Most recent log: {ent_data.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Multi-entity anomalies
    st.markdown("---")
    st.subheader("Entities not seen in last 12 hours")
    anomaly_df = get_all_anomalies(fused, profiles, threshold_hours=MISSING_HOURS_THRESHOLD)
    if anomaly_df.empty:
        st.success("✅ All entities observed within threshold.")
    else:
        st.table(anomaly_df)

    # Export fused timeline
    st.markdown("---")
    st.subheader("Export / Download")
    st.write("You can download the fused timeline for offline analysis.")
    csv = fused.to_csv(index=False)
    st.download_button(label="Download fused timeline (CSV)", data=csv, file_name="fused_timeline.csv", mime="text/csv")

    # Footer / small note
    st.markdown("---")
    st.caption("Prototype — hybrid deterministic/probabilistic ER | For demo purposes only (synthetic data).")


if __name__ == "__main__":
    main()
