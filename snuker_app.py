import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Predictor",
    page_icon="🎱",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #13131f;
        color: #ffffff;
    }
    .stApp { background-color: #13131f; }

    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888888;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 8px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4fc3f7 !important;
        color: #13131f !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1e1e2e !important;
        border: 1px solid #2a2a3a !important;
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Slider */
    .stSlider > div > div > div {
        background-color: #4fc3f7 !important;
    }

    /* Button */
    .stButton > button {
        background-color: #4fc3f7 !important;
        color: #13131f !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 16px;
        border: none;
        padding: 10px 32px;
        border-radius: 4px;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background-color: #81d4fa !important;
        color: #13131f !important;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-cap {
        font-size: 10px;
        letter-spacing: 2px;
        color: #888888;
        margin-bottom: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
    }
    .metric-val-lg {
        font-size: 38px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Player cards */
    .card-a {
        background-color: #1e1e2e;
        border: 2px solid #4fc3f7;
        border-radius: 10px;
        padding: 20px;
    }
    .card-b {
        background-color: #1e1e2e;
        border: 2px solid #ef9a9a;
        border-radius: 10px;
        padding: 20px;
    }
    .card-match {
        background-color: #1e1e2e;
        border: 2px solid #b39ddb;
        border-radius: 10px;
        padding: 20px;
    }

    /* Progress bar */
    .prob-bar-bg {
        background-color: #12121e;
        border-radius: 6px;
        height: 14px;
        width: 100%;
        margin: 8px 0 16px 0;
    }
    .prob-bar-fill-a {
        background-color: #4fc3f7;
        border-radius: 6px;
        height: 14px;
    }
    .prob-bar-fill-b {
        background-color: #ef9a9a;
        border-radius: 6px;
        height: 14px;
    }
    .prob-bar-fill-match {
        background-color: #b39ddb;
        border-radius: 6px;
        height: 14px;
    }

    /* Breakdown rows */
    .breakdown-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #888888;
        padding: 4px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .breakdown-val {
        color: #ffffff;
    }
    .divider {
        border: none;
        border-top: 1px solid #2a2a3a;
        margin: 10px 0;
    }
    .section-cap {
        font-size: 9px;
        letter-spacing: 2px;
        color: #444455;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 6px;
    }
    .match-info {
        text-align: center;
        color: #555566;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        padding: 8px 0;
    }
    .player-name-a {
        font-size: 22px;
        font-weight: 700;
        color: #4fc3f7;
        font-family: 'Rajdhani', sans-serif;
        margin-bottom: 4px;
    }
    .player-name-b {
        font-size: 22px;
        font-weight: 700;
        color: #ef9a9a;
        font-family: 'Rajdhani', sans-serif;
        margin-bottom: 4px;
    }
    .player-name-match {
        font-size: 22px;
        font-weight: 700;
        color: #b39ddb;
        font-family: 'Rajdhani', sans-serif;
        margin-bottom: 4px;
    }

    /* Century distribution table */
    .cen-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        margin-top: 8px;
    }
    .cen-table th {
        color: #444455;
        font-size: 9px;
        letter-spacing: 2px;
        padding: 4px 8px;
        text-align: right;
        border-bottom: 1px solid #2a2a3a;
    }
    .cen-table th:first-child { text-align: left; }
    .cen-table td {
        padding: 4px 8px;
        color: #888888;
        text-align: right;
        border-bottom: 1px solid #1a1a2a;
    }
    .cen-table td:first-child { text-align: left; color: #ffffff; }
    .cen-table tr:last-child td { border-bottom: none; }

    /* Over/Under cards */
    .ou-card {
        background-color: #12121e;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 6px 0;
    }
    .ou-label {
        font-size: 9px;
        letter-spacing: 2px;
        color: #444455;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 2px;
    }
    .ou-prob {
        font-size: 20px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
    }
    .ou-odds {
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
        color: #888888;
        margin-top: 2px;
    }

    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS — Match prediction
# ════════════════════════════════════════════════════════════════════

def elo_expected_frame(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def p_win_match(p_frame, best_of):
    frames_to_win = (best_of // 2) + 1
    total = 0.0
    for wins in range(frames_to_win, best_of + 1):
        total += binom.pmf(frames_to_win - 1, wins - 1, p_frame) * p_frame
    return total

def predict_elob(pa, pb, first_to, ratings):
    best_of = (first_to * 2) - 1
    ra  = ratings.get(pa, {}).get("rating", 0)
    rb  = ratings.get(pb, {}).get("rating", 0)
    pfa = elo_expected_frame(ra, rb)
    pma = p_win_match(pfa, best_of)
    return ra, rb, pfa, 1 - pfa, pma, 1 - pma

def predict_elof(pa, pb, first_to, ratings):
    best_of = (first_to * 2) - 1
    ra  = ratings.get(pa, {}).get("rating", 0)
    rb  = ratings.get(pb, {}).get("rating", 0)
    pfa = elo_expected_frame(ra, rb)
    pma = p_win_match(pfa, best_of)
    return ra, rb, pfa, 1 - pfa, pma, 1 - pma

def predict_match(pa, pb, first_to, elob_w, edge, ratings_elob, ratings_elof):
    best_of = (first_to * 2) - 1
    elof_w  = 1 - elob_w

    bra, brb, bpfa, bpfb, bpma, bpmb = predict_elob(pa, pb, first_to, ratings_elob)
    fra, frb, fpfa, fpfb, fpma, fpmb = predict_elof(pa, pb, first_to, ratings_elof)

    prob_a = elob_w * bpma + elof_w * fpma
    prob_b = 1 - prob_a

    # Blended frame probability
    pfa_blended = elob_w * bpfa + elof_w * fpfa

    return dict(
        prob_a=prob_a,           prob_b=prob_b,
        true_a=1/prob_a,         true_b=1/prob_b,
        edge_a=(1/prob_a)*(1+edge), edge_b=(1/prob_b)*(1+edge),
        elob_ra=bra,  elob_rb=brb,
        elob_pfa=bpfa, elob_pfb=bpfb,
        elob_pa=bpma,  elob_pb=bpmb,
        elof_ra=fra,  elof_rb=frb,
        elof_pfa=fpfa, elof_pfb=fpfb,
        elof_pa=fpma,  elof_pb=fpmb,
        elob_ma=ratings_elob.get(pa, {}).get("matches_played", 0),
        elob_mb=ratings_elob.get(pb, {}).get("matches_played", 0),
        elof_ma=ratings_elof.get(pa, {}).get("matches_played", 0),
        elof_mb=ratings_elof.get(pb, {}).get("matches_played", 0),
        best_of=best_of,
        pfa_blended=pfa_blended,
    )


# ════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS — Century prediction
# ════════════════════════════════════════════════════════════════════

def _match_scoreline_probs(prob_a: float, first_to: int) -> pd.DataFrame:
    scorelines = []
    for p2_wins in range(first_to):
        p1_wins = first_to
        total_frames = p1_wins + p2_wins - 1
        prob = math.comb(total_frames, p2_wins) * (prob_a ** p1_wins) * ((1 - prob_a) ** p2_wins)
        scorelines.append((p1_wins, p2_wins, prob))
    for p1_wins in range(first_to):
        p2_wins = first_to
        total_frames = p1_wins + p2_wins - 1
        prob = math.comb(total_frames, p1_wins) * (prob_a ** p1_wins) * ((1 - prob_a) ** p2_wins)
        scorelines.append((p1_wins, p2_wins, prob))
    return pd.DataFrame(scorelines, columns=["p1_frames", "p2_frames", "prob"])


def _century_distribution(scoreline_df: pd.DataFrame, cen_rate: float) -> dict:
    dist: dict[int, float] = {}
    for _, row in scoreline_df.iterrows():
        p1_wins = int(row["p1_frames"])
        scoreline_prob = row["prob"]
        for centuries in range(p1_wins + 1):
            century_prob = (
                math.comb(p1_wins, centuries)
                * (cen_rate ** centuries)
                * ((1 - cen_rate) ** (p1_wins - centuries))
            )
            dist[centuries] = dist.get(centuries, 0.0) + scoreline_prob * century_prob
    return dict(sorted(dist.items()))


def _match_century_distribution(scoreline_df: pd.DataFrame, cen_rate1: float, cen_rate2: float) -> dict:
    dist: dict[int, float] = {}
    for _, row in scoreline_df.iterrows():
        p1_wins = int(row["p1_frames"])
        p2_wins = int(row["p2_frames"])
        scoreline_prob = row["prob"]
        for c1 in range(p1_wins + 1):
            p_c1 = (
                math.comb(p1_wins, c1)
                * (cen_rate1 ** c1)
                * ((1 - cen_rate1) ** (p1_wins - c1))
            )
            for c2 in range(p2_wins + 1):
                p_c2 = (
                    math.comb(p2_wins, c2)
                    * (cen_rate2 ** c2)
                    * ((1 - cen_rate2) ** (p2_wins - c2))
                )
                total = c1 + c2
                dist[total] = dist.get(total, 0.0) + scoreline_prob * p_c1 * p_c2
    return dict(sorted(dist.items()))


def predict_match_centuries(
    player1_name: str,
    player2_name: str,
    first_to: int,
    player_rates: dict,
    p1_frame_win_prob: float,
) -> dict:
    cen_rate1 = player_rates.get(player1_name, 0.0) or 0.0
    cen_rate2 = player_rates.get(player2_name, 0.0) or 0.0

    scorelines = _match_scoreline_probs(p1_frame_win_prob, first_to)
    p2_scorelines = _match_scoreline_probs(1 - p1_frame_win_prob, first_to)

    dist1 = _century_distribution(scorelines, cen_rate1)
    dist2 = _century_distribution(p2_scorelines, cen_rate2)
    match_dist = _match_century_distribution(scorelines, cen_rate1, cen_rate2)

    return {
        player1_name: dist1,
        player2_name: dist2,
        "match": match_dist,
    }


def over_under(result: dict, selection: str, line: float) -> dict:
    dist = result[selection]
    threshold = int(line - 0.5)
    under = sum(prob for centuries, prob in dist.items() if centuries <= threshold)
    over  = sum(prob for centuries, prob in dist.items() if centuries > threshold)
    return {"under": under, "over": over}


# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def load_ratings():
    with open("ratings.json", encoding="utf-8") as f:
        data = json.load(f)
    return data["elob"], data["elof"]

@st.cache_data
def load_player_rates():
    with open("player_rates.json", encoding="utf-8") as f:
        return json.load(f)

ratings_elob, ratings_elof = load_ratings()
player_rates = load_player_rates()

sorted_players = sorted(
    ratings_elob.keys(),
    key=lambda p: ratings_elob[p]["rating"],
    reverse=True
)


# ════════════════════════════════════════════════════════════════════
# HEADER + SHARED CONTROLS
# ════════════════════════════════════════════════════════════════════

st.markdown("## 🎱 Match Predictor")
st.markdown("---")

col_a, col_vs, col_b = st.columns([10, 1, 10])
with col_a:
    player_a = st.selectbox("PLAYER A", sorted_players, index=0)
with col_vs:
    st.markdown("<br><br>**vs**", unsafe_allow_html=True)
with col_b:
    player_b = st.selectbox("PLAYER B", sorted_players, index=1)

st.markdown("<br>", unsafe_allow_html=True)

sl1, sl2, sl3 = st.columns(3)
with sl1:
    first_to = st.slider("FIRST TO (frames)", 1, 18, 10, step=1)
with sl2:
    elob_w = st.slider("ELOb WEIGHT", 0.0, 1.0, 0.8, step=0.05,
                       format="%.2f",
                       help="Weight given to ELO-beta vs ELO-frames")
with sl3:
    edge_pct = st.slider("EDGE TARGET", 0.0, 20.0, 5.0, step=0.5, format="%.1f%%")
    edge = edge_pct / 100

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("RUN PREDICTION")

# Compute shared result once and store in session state
if run:
    if player_a == player_b:
        st.error("Please select two different players.")
        st.stop()
    st.session_state["result"] = predict_match(
        player_a, player_b, first_to,
        elob_w, edge, ratings_elob, ratings_elof
    )
    st.session_state["players"] = (player_a, player_b)
    st.session_state["first_to"] = first_to
    st.session_state["edge"] = edge


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════

tab_match, tab_centuries = st.tabs(["📊  MATCH ODDS", "🔴  CENTURIES"])


# ────────────────────────────────────────────────────────────────────
# TAB 1 — Match Odds (original display)
# ────────────────────────────────────────────────────────────────────

def render_card(col, name, prob, true_o, edge_p,
                elob_r, elof_r,
                elob_pf, elob_pm,
                elof_pf, elof_pm,
                m_elob, m_elof,
                color, bar_class, card_class, edge_pct_val):
    with col:
        bar_w = int(prob * 100)
        name_class = "player-name-a" if color == "#4fc3f7" else "player-name-b"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="{name_class}">{name}</div>
            <div class="prob-bar-bg">
                <div class="{bar_class}" style="width:{bar_w}%;"></div>
            </div>
            <div style="display:flex; gap:8px; margin-bottom:8px;">
                <div class="metric-card" style="flex:1;">
                    <div class="metric-cap">WIN PROBABILITY</div>
                    <div class="metric-val" style="color:{color};">{prob*100:.1f}%</div>
                </div>
                <div class="metric-card" style="flex:1;">
                    <div class="metric-cap">TRUE ODDS</div>
                    <div class="metric-val">{true_o:.3f}</div>
                </div>
            </div>
            <div class="metric-card" style="margin-bottom:12px;">
                <div class="metric-cap">TARGET (+{edge_pct_val:.1%} EDGE)</div>
                <div class="metric-val-lg" style="color:{color};">{edge_p:.3f}</div>
            </div>
            <hr class="divider">
            <div class="section-cap">MODEL BREAKDOWN</div>
            <div class="breakdown-row">
                <span>ELO-beta (rating: {elob_r:+.1f})</span>
                <span class="breakdown-val">frame / match = {elob_pf*100:.1f}% / {elob_pm*100:.1f}%</span>
            </div>
            <div class="breakdown-row">
                <span>ELO-frames (rating: {elof_r:+.1f})</span>
                <span class="breakdown-val">frame / match = {elof_pf*100:.1f}% / {elof_pm*100:.1f}%</span>
            </div>
            <hr class="divider">
            <div class="breakdown-row">
                <span>Matches (β / f)</span>
                <span class="breakdown-val">{m_elob} / {m_elof}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_match:
    if "result" not in st.session_state:
        st.markdown('<div class="match-info">Select players and press RUN PREDICTION</div>', unsafe_allow_html=True)
    else:
        r = st.session_state["result"]
        pa, pb = st.session_state["players"]
        st.markdown(
            f'<div class="match-info">'
            f'First to {st.session_state["first_to"]} &nbsp;·&nbsp; Best of {r["best_of"]} &nbsp;·&nbsp; '
            f'ELOb {elob_w:.0%} / ELOf {1-elob_w:.0%}'
            f'</div>',
            unsafe_allow_html=True
        )
        card_a, card_b = st.columns(2)
        render_card(card_a, pa, r["prob_a"], r["true_a"], r["edge_a"],
                    r["elob_ra"], r["elof_ra"], r["elob_pfa"], r["elob_pa"],
                    r["elof_pfa"], r["elof_pa"], r["elob_ma"], r["elof_ma"],
                    "#4fc3f7", "prob-bar-fill-a", "card-a", st.session_state["edge"])
        render_card(card_b, pb, r["prob_b"], r["true_b"], r["edge_b"],
                    r["elob_rb"], r["elof_rb"], r["elob_pfb"], r["elob_pb"],
                    r["elof_pfb"], r["elof_pb"], r["elob_mb"], r["elof_mb"],
                    "#ef9a9a", "prob-bar-fill-b", "card-b", st.session_state["edge"])


# ────────────────────────────────────────────────────────────────────
# TAB 2 — Centuries
# ────────────────────────────────────────────────────────────────────

def _ou_block(label, color, ou, line, cen_edge):
    """Render an over/under block with true and target odds."""
    over_true  = 1 / ou["over"]  if ou["over"]  > 0 else float("inf")
    under_true = 1 / ou["under"] if ou["under"] > 0 else float("inf")
    over_tgt   = over_true  * (1 + cen_edge)
    under_tgt  = under_true * (1 + cen_edge)
    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <div style="font-size:10px; letter-spacing:2px; color:#444455;
                    font-family:'IBM Plex Mono',monospace; margin-bottom:6px;">
            LINE &nbsp;{line}
        </div>
        <div style="display:flex; gap:8px;">
            <div class="ou-card" style="flex:1; border-left: 3px solid {color};">
                <div class="ou-label">OVER {line}</div>
                <div class="ou-prob" style="color:{color};">{ou['over']*100:.1f}%</div>
                <div class="ou-odds">true {over_true:.3f} &nbsp;·&nbsp; tgt {over_tgt:.3f}</div>
            </div>
            <div class="ou-card" style="flex:1; border-left: 3px solid #444455;">
                <div class="ou-label">UNDER {line}</div>
                <div class="ou-prob" style="color:#888888;">{ou['under']*100:.1f}%</div>
                <div class="ou-odds">true {under_true:.3f} &nbsp;·&nbsp; tgt {under_tgt:.3f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _dist_table(dist: dict, color: str):
    """Render a compact probability distribution table."""
    rows = ""
    for k, v in dist.items():
        bar_w = int(v * 200)
        bar_w = min(bar_w, 100)
        rows += f"""
        <tr>
            <td>{k}</td>
            <td>
                <div style="background:#12121e;border-radius:3px;height:8px;width:100%;">
                    <div style="background:{color};border-radius:3px;height:8px;width:{bar_w}%;"></div>
                </div>
            </td>
            <td>{v*100:.2f}%</td>
            <td>{1/v:.2f}</td>
        </tr>
        """
    st.markdown(f"""
    <table class="cen-table">
        <thead>
            <tr>
                <th>CENTURIES</th>
                <th style="width:40%;">DIST</th>
                <th>PROB</th>
                <th>TRUE</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


with tab_centuries:
    if "result" not in st.session_state:
        st.markdown('<div class="match-info">Select players and press RUN PREDICTION</div>', unsafe_allow_html=True)
    else:
        r   = st.session_state["result"]
        pa, pb = st.session_state["players"]
        ft  = st.session_state["first_to"]
        cen_edge_pct = st.slider("CENTURIES EDGE TARGET", 0.0, 20.0, 5.0,
                                  step=0.5, format="%.1f%%", key="cen_edge")
        cen_edge = cen_edge_pct / 100

        # Max line = first_to (a player can't make more centuries than frames won)
        max_line_val = ft
        line_options = [x / 2 for x in range(1, max_line_val * 2 + 1, 2)]  # 0.5, 1.5, 2.5 ...

        st.markdown("<br>", unsafe_allow_html=True)
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            if len(line_options) == 1:
                line_a = line_options[0]
                st.markdown(f'<div class="match-info">{pa} LINE: {line_a}</div>', unsafe_allow_html=True)
            else:
                line_a = st.select_slider(f"{pa} LINE", options=line_options,
                                          value=line_options[min(1, len(line_options)-1)], key="line_a")
        with lc2:
            if len(line_options) == 1:
                line_b = line_options[0]
                st.markdown(f'<div class="match-info">{pb} LINE: {line_b}</div>', unsafe_allow_html=True)
            else:
                line_b = st.select_slider(f"{pb} LINE", options=line_options,
                                          value=line_options[min(1, len(line_options)-1)], key="line_b")
        with lc3:
            match_line_options = [x / 2 for x in range(1, ft * 4, 2)]  # wider range for match
            if len(match_line_options) == 1:
                line_m = match_line_options[0]
                st.markdown(f'<div class="match-info">MATCH LINE: {line_m}</div>', unsafe_allow_html=True)
            else:
                line_m = st.select_slider("MATCH LINE", options=match_line_options,
                                          value=match_line_options[min(2, len(match_line_options)-1)], key="line_m")

        st.markdown("<br>", unsafe_allow_html=True)

        # Run century model
        cen_result = predict_match_centuries(
            player1_name=pa,
            player2_name=pb,
            first_to=ft,
            player_rates=player_rates,
            p1_frame_win_prob=r["pfa_blended"],
        )

        ou_a = over_under(cen_result, pa, line_a)
        ou_b = over_under(cen_result, pb, line_b)
        ou_m = over_under(cen_result, "match", line_m)

        # Display cards
        col_ca, col_cb, col_cm = st.columns(3)

        with col_ca:
            cen_rate_a = player_rates.get(pa, 0.0) or 0.0
            st.markdown(f"""
            <div class="card-a">
                <div class="player-name-a">{pa}</div>
                <div class="breakdown-row" style="margin-bottom:8px;">
                    <span>century rate (given frame win)</span>
                    <span class="breakdown-val">{cen_rate_a*100:.1f}%</span>
                </div>
                <hr class="divider">
            """, unsafe_allow_html=True)
            _ou_block(pa, "#4fc3f7", ou_a, line_a, cen_edge)
            st.markdown('<hr class="divider"><div class="section-cap">FULL DISTRIBUTION</div>', unsafe_allow_html=True)
            _dist_table(cen_result[pa], "#4fc3f7")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_cb:
            cen_rate_b = player_rates.get(pb, 0.0) or 0.0
            st.markdown(f"""
            <div class="card-b">
                <div class="player-name-b">{pb}</div>
                <div class="breakdown-row" style="margin-bottom:8px;">
                    <span>century rate (given frame win)</span>
                    <span class="breakdown-val">{cen_rate_b*100:.1f}%</span>
                </div>
                <hr class="divider">
            """, unsafe_allow_html=True)
            _ou_block(pb, "#ef9a9a", ou_b, line_b, cen_edge)
            st.markdown('<hr class="divider"><div class="section-cap">FULL DISTRIBUTION</div>', unsafe_allow_html=True)
            _dist_table(cen_result[pb], "#ef9a9a")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_cm:
            st.markdown(f"""
            <div class="card-match">
                <div class="player-name-match">MATCH TOTAL</div>
                <div class="breakdown-row" style="margin-bottom:8px;">
                    <span>combined centuries</span>
                    <span class="breakdown-val">{pa} + {pb}</span>
                </div>
                <hr class="divider">
            """, unsafe_allow_html=True)
            _ou_block("match", "#b39ddb", ou_m, line_m, cen_edge)
            st.markdown('<hr class="divider"><div class="section-cap">FULL DISTRIBUTION</div>', unsafe_allow_html=True)
            _dist_table(cen_result["match"], "#b39ddb")
            st.markdown("</div>", unsafe_allow_html=True)
