import os
from datetime import date

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:9696").rstrip("/")

RED = "#D20A0A"
BLUE = "#1F4FD8"

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="centered")

st.markdown(
    f"""
    <style>
    .title {{ font-size: 2.6rem; font-weight: 900; text-transform: uppercase;
              letter-spacing: 2px; margin-bottom: 0; }}
    .title span {{ color: {RED}; }}
    .subtitle {{ color: #9a9a9a; margin-top: 0; }}
    .corner {{ padding: 0.6rem 1rem; border-radius: 6px; font-weight: 800;
               text-transform: uppercase; letter-spacing: 1px; text-align: center; }}
    .red {{ background: {RED}; }}
    .blue {{ background: {BLUE}; }}
    .vs {{ font-size: 2rem; font-weight: 900; text-align: center; padding-top: 1.6rem; }}
    .prob {{ font-size: 2.4rem; font-weight: 900; text-align: center; margin: 0; }}
    .name {{ text-align: center; color: #cfcfcf; margin: 0; }}
    .bar {{ display: flex; height: 18px; border-radius: 9px; overflow: hidden; margin: 0.8rem 0; }}
    .winner {{ text-align: center; font-size: 1.3rem; font-weight: 800; margin-top: 0.5rem; }}
    .winner span {{ color: #D4AF37; }}
    .footer {{ color: #6b6b6b; font-size: 0.8rem; text-align: center; margin-top: 3rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Loading fighters...")
def load_fighters():
    resp = requests.get(f"{API_URL}/fighters", timeout=90)   # free cloud tiers may need to wake up
    resp.raise_for_status()
    return {f["FIGHTER"]: f for f in resp.json()}

@st.cache_data(ttl=3600)
def load_data_until():
    resp = requests.get(f"{API_URL}/health", timeout=90)
    resp.raise_for_status()
    return resp.json().get("data_until")


st.markdown('<div class="title">Fight <span>Predictor</span></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Pick two UFC fighters and get the win probability of each one.</p>',
            unsafe_allow_html=True)

try:
    fighters = load_fighters()
except requests.RequestException as e:
    st.error(f"Could not reach the prediction API at {API_URL}: {e}")
    st.stop()

try:
    data_until = load_data_until()
except requests.RequestException:
    data_until = None

if data_until:
    pretty = date.fromisoformat(data_until).strftime("%d %b %Y")
    st.caption(f"📅 Fighter statistics include UFC fights up to **{pretty}**. "
               "Fights after that date are not reflected in the predictions.")

names = sorted(fighters)

# --- Fighter selection -------------------------------------------------------
col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    st.markdown('<div class="corner red">Red corner</div>', unsafe_allow_html=True)
    fighter_a = st.selectbox("Fighter A", names, index=None,
                             placeholder="Type a name...", label_visibility="collapsed")
with col_vs:
    st.markdown('<div class="vs">VS</div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="corner blue">Blue corner</div>', unsafe_allow_html=True)
    if fighter_a:
        divs_a = set(fighters[fighter_a]["divisions"])
        opponents = [n for n in names if n != fighter_a and divs_a & set(fighters[n]["divisions"])]
    else:
        opponents = names
    fighter_b = st.selectbox("Fighter B", opponents, index=None,
                             placeholder="Type a name...", label_visibility="collapsed")

# --- Fight context -------------------------------------------------------------
st.markdown("#### Fight details")
if fighter_a and fighter_b:
    shared = sorted(set(fighters[fighter_a]["divisions"]) & set(fighters[fighter_b]["divisions"]))
elif fighter_a:
    shared = fighters[fighter_a]["divisions"]
else:
    shared = []
last_a = fighters[fighter_a]["weight_class"] if fighter_a else None

c1, c2, c3 = st.columns([3, 2, 2])
with c1:
    weight_class = st.selectbox("Weight class", shared,
                                index=shared.index(last_a) if last_a in shared else 0,
                                disabled=not shared, placeholder="Select the fighters first")
with c2:
    is_title = st.checkbox("Title fight")
with c3:
    rounds = st.radio("Rounds", [3, 5], index=1 if is_title else 0, horizontal=True)

# --- Prediction ----------------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):
    if not fighter_a or not fighter_b:
        st.warning("Select both fighters.")
    elif fighter_a == fighter_b:
        st.warning("Select two different fighters.")
    else:
        payload = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "weight_class": weight_class,
            "is_title": int(is_title),
            "scheduled_rounds": rounds,
        }
        with st.spinner("Analysing the matchup..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        p_a = result["fighter_a_win_prob"]
        p_b = result["fighter_b_win_prob"]

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f'<div class="prob" style="color:{RED}">{p_a:.0%}</div>'
                        f'<p class="name">{fighter_a}</p>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="prob" style="color:{BLUE}">{p_b:.0%}</div>'
                        f'<p class="name">{fighter_b}</p>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="bar"><div style="width:{p_a*100:.1f}%;background:{RED}"></div>'
            f'<div style="width:{p_b*100:.1f}%;background:{BLUE}"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="winner">🏆 Predicted winner: <span>{result["predicted_winner"]}</span></div>',
                    unsafe_allow_html=True)

        for w in result.get("warnings", []):
            st.warning(w)

st.markdown('<p class="footer">ML Zoomcamp project · Data from UFC Stats · '
            'Not affiliated with or endorsed by the UFC</p>', unsafe_allow_html=True)