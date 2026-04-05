import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Match Predictor", page_icon="🎱", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #13131f; color: #ffffff; }
    .stApp { background-color: #13131f; }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e1e2e; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #888888; font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 15px; letter-spacing: 1px; border-radius: 6px; padding: 8px 24px; }
    .stTabs [aria-selected="true"] { background-color: #4fc3f7 !important; color: #13131f !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }
    .stSelectbox > div > div { background-color: #1e1e2e !important; border: 1px solid #2a2a3a !important; color: #ffffff !important; font-family: 'Rajdhani', sans-serif; }
    .stSlider > div > div > div { background-color: #4fc3f7 !important; }
    .stButton > button { background-color: #4fc3f7 !important; color: #13131f !important; font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 16px; border: none; padding: 10px 32px; border-radius: 4px; letter-spacing: 1px; }
    .stButton > button:hover { background-color: #81d4fa !important; color: #13131f !important; }
    .stTextInput > div > div > input { background-color: #1e1e2e !important; border: 1px solid #2a2a3a !important; color: #ffffff !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; padding: 4px 8px !important; border-radius: 4px !important; }
    .metric-card { background-color: #1e1e2e; border-radius: 8px; padding: 16px; text-align: center; margin: 4px 0; }
    .metric-cap { font-size: 10px; letter-spacing: 2px; color: #888888; margin-bottom: 4px; font-family: 'IBM Plex Mono', monospace; }
    .metric-val { font-size: 32px; font-weight: 700; font-family: 'Rajdhani', sans-serif; }
    .metric-val-lg { font-size: 38px; font-weight: 700; font-family: 'Rajdhani', sans-serif; }
    .card-a { background-color: #1e1e2e; border: 2px solid #4fc3f7; border-radius: 10px; padding: 20px; }
    .card-b { background-color: #1e1e2e; border: 2px solid #ef9a9a; border-radius: 10px; padding: 20px; }
    .card-match { background-color: #1e1e2e; border: 2px solid #b39ddb; border-radius: 10px; padding: 20px; }
    .prob-bar-bg { background-color: #12121e; border-radius: 6px; height: 14px; width: 100%; margin: 8px 0 16px 0; }
    .prob-bar-fill-a { background-color: #4fc3f7; border-radius: 6px; height: 14px; }
    .prob-bar-fill-b { background-color: #ef9a9a; border-radius: 6px; height: 14px; }
    .breakdown-row { display: flex; justify-content: space-between; font-size: 13px; color: #888888; padding: 4px 0; font-family: 'IBM Plex Mono', monospace; }
    .breakdown-val { color: #ffffff; }
    .divider { border: none; border-top: 1px solid #2a2a3a; margin: 10px 0; }
    .section-cap { font-size: 9px; letter-spacing: 2px; color: #444455; font-family: 'IBM Plex Mono', monospace; margin-bottom: 6px; }
    .match-info { text-align: center; color: #555566; font-size: 12px; font-family: 'IBM Plex Mono', monospace; padding: 8px 0; }
    .player-name-a { font-size: 22px; font-weight: 700; color: #4fc3f7; font-family: 'Rajdhani', sans-serif; margin-bottom: 4px; }
    .player-name-b { font-size: 22px; font-weight: 700; color: #ef9a9a; font-family: 'Rajdhani', sans-serif; margin-bottom: 4px; }
    .player-name-match { font-size: 22px; font-weight: 700; color: #b39ddb; font-family: 'Rajdhani', sans-serif; margin-bottom: 4px; }
    .cen-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 12px; margin-top: 8px; }
    .cen-table th { color: #444455; font-size: 9px; letter-spacing: 2px; padding: 4px 8px; text-align: right; border-bottom: 1px solid #2a2a3a; }
    .cen-table th:first-child { text-align: left; }
    .cen-table td { padding: 4px 8px; color: #888888; text-align: right; border-bottom: 1px solid #1a1a2a; }
    .cen-table td:first-child { text-align: left; color: #ffffff; }
    .cen-table tr:last-child td { border-bottom: none; }
    .ou-card { background-color: #12121e; border-radius: 8px; padding: 14px 16px; margin: 6px 0; }
    .ou-label { font-size: 9px; letter-spacing: 2px; color: #444455; font-family: 'IBM Plex Mono', monospace; margin-bottom: 2px; }
    .ou-prob { font-size: 20px; font-weight: 700; font-family: 'Rajdhani', sans-serif; }
    .ou-odds { font-size: 12px; font-family: 'IBM Plex Mono', monospace; color: #888888; margin-top: 2px; }
    .bet-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 12px; margin-top: 8px; }
    .bet-table th { color: #444455; font-size: 9px; letter-spacing: 2px; padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2a3a; }
    .bet-table td { padding: 8px 12px; color: #cccccc; text-align: left; border-bottom: 1px solid #1a1a2a; }
    .bet-table tr:last-child td { border-bottom: none; }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════

if "bet_slip" not in st.session_state:
    st.session_state["bet_slip"] = pd.DataFrame(
        columns=["Player 1", "Player 2", "Market", "Selection", "Line",
                 "Target", "Market Price", "Kelly Stake %"]
    )
if "mp_version" not in st.session_state:
    st.session_state["mp_version"] = 0   # bump this to wipe all market price inputs


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def kelly_stake(market_odds: float, prob_win: float) -> float:
    """Kelly criterion as a fraction of bankroll (capped at 0)."""
    b = market_odds - 1
    q = 1 - prob_win
    if b <= 0:
        return 0.0
    k = (b * prob_win - q) / b
    return max(k, 0.0)


def add_bet(player1, player2, market, selection, line, target, market_price):
    prob_win = 1 / target if target > 0 else 0.0
    k = kelly_stake(market_price, prob_win)
    new_row = pd.DataFrame([{
        "Player 1": player1, "Player 2": player2, "Market": market,
        "Selection": selection, "Line": line,
        "Target": round(float(target), 3),
        "Market Price": round(float(market_price), 3),
        "Kelly Stake %": f"{k*100:.2f}%",
    }])
    st.session_state["bet_slip"] = pd.concat(
        [st.session_state["bet_slip"], new_row], ignore_index=True
    )


def market_input_row(key_prefix, target_val, pa, pb, market, selection, line):
    """Renders market price text input + Add Bet button. Returns (mp|None, is_value)."""
    versioned_key = f"mp_{key_prefix}_v{st.session_state['mp_version']}"
    c1, c2 = st.columns([4, 1])
    with c1:
        raw = st.text_input("mkt", value="", key=versioned_key,
                            label_visibility="collapsed", placeholder="Market price")
    with c2:
        btn = st.button("＋", key=f"ab_{key_prefix}_v{st.session_state['mp_version']}",
                        help="Add to bet slip")

    mp_val, is_value = None, False
    if raw.strip():
        try:
            mp_val = float(raw.strip())
            is_value = mp_val > target_val
        except ValueError:
            pass

    if btn and mp_val is not None and is_value:
        add_bet(pa, pb, market, selection, line, target_val, mp_val)
        st.toast(f"Added: {selection} @ {mp_val:.3f}", icon="✅")

    return mp_val, is_value


def value_badge(mp_val, target_val, is_value):
    if mp_val is None:
        return ""
    c = "#a5d6a7" if is_value else "#ef9a9a"
    icon = "✅ value" if is_value else "❌ no value"
    return (f"<div style='font-size:12px;font-weight:700;color:{c};"
            f"font-family:monospace;margin:2px 0 6px 0;'>"
            f"{mp_val:.3f} &nbsp;{icon}</div>")


def collapse_dist(dist: dict, threshold: float = 0.02) -> list:
    """
    Returns a list of (label, prob) collapsing any tail entries below threshold
    into a single 'N+' bucket.
    """
    items = sorted(dist.items())
    result = []
    tail_prob = 0.0
    tail_start = None
    for k, v in items:
        if v < threshold:
            if tail_start is None:
                tail_start = k
            tail_prob += v
        else:
            result.append((str(k), v))
    if tail_prob > 0 and tail_start is not None:
        result.append((f"{tail_start}+", tail_prob))
    return result


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
    ra = ratings.get(pa, {}).get("rating", 0)
    rb = ratings.get(pb, {}).get("rating", 0)
    pfa = elo_expected_frame(ra, rb)
    pma = p_win_match(pfa, best_of)
    return ra, rb, pfa, 1 - pfa, pma, 1 - pma

def predict_elof(pa, pb, first_to, ratings):
    best_of = (first_to * 2) - 1
    ra = ratings.get(pa, {}).get("rating", 0)
    rb = ratings.get(pb, {}).get("rating", 0)
    pfa = elo_expected_frame(ra, rb)
    pma = p_win_match(pfa, best_of)
    return ra, rb, pfa, 1 - pfa, pma, 1 - pma

def predict_match(pa, pb, first_to, elob_w, edge, ratings_elob, ratings_elof):
    best_of = (first_to * 2) - 1
    elof_w = 1 - elob_w
    bra, brb, bpfa, bpfb, bpma, bpmb = predict_elob(pa, pb, first_to, ratings_elob)
    fra, frb, fpfa, fpfb, fpma, fpmb = predict_elof(pa, pb, first_to, ratings_elof)
    prob_a = elob_w * bpma + elof_w * fpma
    prob_b = 1 - prob_a
    pfa_blended = elob_w * bpfa + elof_w * fpfa
    return dict(
        prob_a=prob_a, prob_b=prob_b,
        true_a=1/prob_a, true_b=1/prob_b,
        edge_a=(1/prob_a)*(1+edge), edge_b=(1/prob_b)*(1+edge),
        elob_ra=bra, elob_rb=brb, elob_pfa=bpfa, elob_pfb=bpfb, elob_pa=bpma, elob_pb=bpmb,
        elof_ra=fra, elof_rb=frb, elof_pfa=fpfa, elof_pfb=fpfb, elof_pa=fpma, elof_pb=fpmb,
        elob_ma=ratings_elob.get(pa, {}).get("matches_played", 0),
        elob_mb=ratings_elob.get(pb, {}).get("matches_played", 0),
        elof_ma=ratings_elof.get(pa, {}).get("matches_played", 0),
        elof_mb=ratings_elof.get(pb, {}).get("matches_played", 0),
        best_of=best_of, pfa_blended=pfa_blended,
    )


# ════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS — Scoreline & Handicap
# ════════════════════════════════════════════════════════════════════

def scoreline_probs(prob_a: float, first_to: int) -> dict:
    probs = {}
    for p2_wins in range(first_to):
        p1_wins = first_to
        tf = p1_wins + p2_wins - 1
        probs[(p1_wins, p2_wins)] = math.comb(tf, p2_wins) * (prob_a**p1_wins) * ((1-prob_a)**p2_wins)
    for p1_wins in range(first_to):
        p2_wins = first_to
        tf = p1_wins + p2_wins - 1
        probs[(p1_wins, p2_wins)] = math.comb(tf, p1_wins) * (prob_a**p1_wins) * ((1-prob_a)**p2_wins)
    return probs

def handicap_prob_p1(scorelines, handicap):
    return sum(p for (p1, p2), p in scorelines.items() if (p1 + handicap) > p2)

def handicap_prob_p2(scorelines, handicap):
    return sum(p for (p1, p2), p in scorelines.items() if (p2 + handicap) > p1)


# ════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS — Century prediction
# ════════════════════════════════════════════════════════════════════

def _match_scoreline_df(prob_a, first_to):
    rows = []
    for p2_wins in range(first_to):
        p1_wins = first_to
        tf = p1_wins + p2_wins - 1
        rows.append((p1_wins, p2_wins, math.comb(tf, p2_wins) * (prob_a**p1_wins) * ((1-prob_a)**p2_wins)))
    for p1_wins in range(first_to):
        p2_wins = first_to
        tf = p1_wins + p2_wins - 1
        rows.append((p1_wins, p2_wins, math.comb(tf, p1_wins) * (prob_a**p1_wins) * ((1-prob_a)**p2_wins)))
    return pd.DataFrame(rows, columns=["p1_frames", "p2_frames", "prob"])

def _century_distribution(scoreline_df, cen_rate):
    dist = {}
    for _, row in scoreline_df.iterrows():
        p1w = int(row["p1_frames"])
        sp = row["prob"]
        for c in range(p1w + 1):
            cp = math.comb(p1w, c) * (cen_rate**c) * ((1-cen_rate)**(p1w-c))
            dist[c] = dist.get(c, 0.0) + sp * cp
    return dict(sorted(dist.items()))

def _match_century_distribution(scoreline_df, cen_rate1, cen_rate2):
    dist = {}
    for _, row in scoreline_df.iterrows():
        p1w, p2w, sp = int(row["p1_frames"]), int(row["p2_frames"]), row["prob"]
        for c1 in range(p1w + 1):
            pc1 = math.comb(p1w, c1) * (cen_rate1**c1) * ((1-cen_rate1)**(p1w-c1))
            for c2 in range(p2w + 1):
                pc2 = math.comb(p2w, c2) * (cen_rate2**c2) * ((1-cen_rate2)**(p2w-c2))
                t = c1 + c2
                dist[t] = dist.get(t, 0.0) + sp * pc1 * pc2
    return dict(sorted(dist.items()))

def predict_match_centuries(player1_name, player2_name, first_to, player_rates, p1_frame_win_prob):
    cr1 = player_rates.get(player1_name, 0.0) or 0.0
    cr2 = player_rates.get(player2_name, 0.0) or 0.0
    sl  = _match_scoreline_df(p1_frame_win_prob, first_to)
    sl2 = _match_scoreline_df(1 - p1_frame_win_prob, first_to)
    return {
        player1_name: _century_distribution(sl, cr1),
        player2_name: _century_distribution(sl2, cr2),
        "match": _match_century_distribution(sl, cr1, cr2),
    }

def over_under(result, selection, line):
    dist = result[selection]
    threshold = int(line - 0.5)
    under = sum(p for c, p in dist.items() if c <= threshold)
    over  = sum(p for c, p in dist.items() if c > threshold)
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
sorted_players = sorted(ratings_elob.keys(), key=lambda p: ratings_elob[p]["rating"], reverse=True)


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
    elob_w = st.slider("ELOb WEIGHT", 0.0, 1.0, 0.8, step=0.05, format="%.2f",
                       help="Weight given to ELO-beta vs ELO-frames")
with sl3:
    edge_pct = st.slider("EDGE TARGET", 0.0, 20.0, 5.0, step=0.5, format="%.1f%%")
    edge = edge_pct / 100

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("RUN PREDICTION")

if run:
    if player_a == player_b:
        st.error("Please select two different players.")
        st.stop()
    st.session_state["result"] = predict_match(
        player_a, player_b, first_to, elob_w, edge, ratings_elob, ratings_elof)
    st.session_state["players"] = (player_a, player_b)
    st.session_state["first_to"] = first_to
    st.session_state["edge"] = edge
    # Bump version to wipe all market price inputs across all tabs
    st.session_state["mp_version"] = st.session_state.get("mp_version", 0) + 1


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════

tab_match, tab_handicap, tab_centuries, tab_betslip, tab_results = st.tabs([
    "📊  MATCH ODDS", "↕️  HANDICAPS", "🔴  CENTURIES", "📋  BET SLIP", "📅  RESULTS",
])


# ────────────────────────────────────────────────────────────────────
# TAB 1 — Match Odds
# ────────────────────────────────────────────────────────────────────

def render_match_card(col, name, prob, true_o, edge_p,
                      elob_r, elof_r, elob_pf, elob_pm, elof_pf, elof_pm,
                      m_elob, h_edge, color, bar_class, card_class, pa, pb, pkey):
    with col:
        bar_w = int(prob * 100)
        name_class = "player-name-a" if color == "#4fc3f7" else "player-name-b"
        matches_color = "#ef5350" if m_elob < 50 else "#ffffff"
        matches_weight = "700" if m_elob < 50 else "400"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="{name_class}">{name}</div>
            <div class="prob-bar-bg"><div class="{bar_class}" style="width:{bar_w}%;"></div></div>
            <div style="display:flex;gap:8px;margin-bottom:8px;">
                <div class="metric-card" style="flex:1;"><div class="metric-cap">WIN PROBABILITY</div><div class="metric-val" style="color:{color};">{prob*100:.1f}%</div></div>
                <div class="metric-card" style="flex:1;"><div class="metric-cap">TRUE ODDS</div><div class="metric-val">{true_o:.3f}</div></div>
            </div>
            <div class="metric-card" style="margin-bottom:12px;">
                <div class="metric-cap">TARGET (+{h_edge:.1%} EDGE)</div>
                <div class="metric-val-lg" style="color:{color};">{edge_p:.3f}</div>
            </div>
            <hr class="divider">
            <div class="section-cap">MODEL BREAKDOWN</div>
            <div class="breakdown-row"><span>ELO-beta (rating: {elob_r:+.1f})</span><span class="breakdown-val">frame / match = {elob_pf*100:.1f}% / {elob_pm*100:.1f}%</span></div>
            <div class="breakdown-row"><span>ELO-frames (rating: {elof_r:+.1f})</span><span class="breakdown-val">frame / match = {elof_pf*100:.1f}% / {elof_pm*100:.1f}%</span></div>
            <hr class="divider">
            <div class="breakdown-row"><span>Matches</span><span style="color:{matches_color};font-weight:{matches_weight};font-family:'IBM Plex Mono',monospace;">{m_elob}</span></div>
        </div>
        """, unsafe_allow_html=True)
        mp_val, is_val = market_input_row(f"mo_{pkey}", edge_p, pa, pb, "Match Odds", name, 0)
        if mp_val is not None:
            tgt_color = "#a5d6a7" if is_val else "#ef9a9a"
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;color:{tgt_color};"
                f"font-family:monospace;margin-top:2px;'>"
                f"{mp_val:.3f} &nbsp;{'✅ value' if is_val else '❌ no value'}</div>",
                unsafe_allow_html=True)

with tab_match:
    if "result" not in st.session_state:
        st.markdown('<div class="match-info">Select players and press RUN PREDICTION</div>', unsafe_allow_html=True)
    else:
        r = st.session_state["result"]
        pa, pb = st.session_state["players"]
        st.markdown(
            f'<div class="match-info">First to {st.session_state["first_to"]} &nbsp;·&nbsp; '
            f'Best of {r["best_of"]} &nbsp;·&nbsp; ELOb {elob_w:.0%} / ELOf {1-elob_w:.0%}</div>',
            unsafe_allow_html=True)
        ca, cb = st.columns(2)
        render_match_card(ca, pa, r["prob_a"], r["true_a"], r["edge_a"],
                          r["elob_ra"], r["elof_ra"], r["elob_pfa"], r["elob_pa"],
                          r["elof_pfa"], r["elof_pa"], r["elob_ma"],
                          st.session_state["edge"], "#4fc3f7", "prob-bar-fill-a", "card-a", pa, pb, "a")
        render_match_card(cb, pb, r["prob_b"], r["true_b"], r["edge_b"],
                          r["elob_rb"], r["elof_rb"], r["elob_pfb"], r["elob_pb"],
                          r["elof_pfb"], r["elof_pb"], r["elob_mb"],
                          st.session_state["edge"], "#ef9a9a", "prob-bar-fill-b", "card-b", pa, pb, "b")


# ────────────────────────────────────────────────────────────────────
# TAB 2 — Handicaps
# ────────────────────────────────────────────────────────────────────

def _hcap_card_html(name, name_class, card_class, color, lines_data):
    rows = ""
    for label, prob, true_o, tgt_o, bar_w in lines_data:
        rows += (
            f"<div style='display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #1a1a2a;'>"
            f"<div style='font-family:monospace;font-size:13px;color:#fff;width:52px;flex-shrink:0;'>{label}</div>"
            f"<div style='flex:1;background:#12121e;border-radius:4px;height:10px;'>"
            f"<div style='background:{color};border-radius:4px;height:10px;width:{bar_w}%;'></div></div>"
            f"<div style='font-size:15px;font-weight:700;color:{color};width:52px;text-align:right;flex-shrink:0;'>{prob*100:.1f}%</div>"
            f"<div style='font-size:16px;font-weight:700;color:{color};width:60px;text-align:right;flex-shrink:0;'>{tgt_o:.3f}</div>"
            f"<div style='font-size:11px;color:#555566;width:54px;text-align:right;flex-shrink:0;'>{true_o:.3f}</div>"
            f"</div>"
        )
    if not rows:
        rows = "<div style='text-align:center;color:#555566;font-size:12px;padding:16px 0;'>No lines in range for this match length.</div>"
    return (
        f"<div class='{card_class}'><div class='{name_class}'>{name}</div><hr class='divider'>"
        f"<div style='display:flex;justify-content:flex-end;gap:8px;margin-bottom:2px;'>"
        f"<span style='font-size:9px;color:{color};font-weight:700;letter-spacing:1px;width:60px;text-align:right;'>TARGET</span>"
        f"<span style='font-size:9px;color:#444455;letter-spacing:1px;width:54px;text-align:right;'>TRUE</span>"
        f"</div>" + rows + "</div>"
    )


with tab_handicap:
    if "result" not in st.session_state:
        st.markdown('<div class="match-info">Select players and press RUN PREDICTION</div>', unsafe_allow_html=True)
    else:
        r = st.session_state["result"]
        pa, pb = st.session_state["players"]
        ft = st.session_state["first_to"]
        h_edge = st.session_state["edge"]

        st.markdown(f'<div class="match-info">First to {ft} &nbsp;·&nbsp; Best of {r["best_of"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sl = scoreline_probs(r["pfa_blended"], ft)
        # Extended to ±5.5
        candidate_lines = [-5.5, -4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5]

        def line_possible(line, ft):
            return ft >= math.ceil(abs(line))

        def build_lines(prob_fn):
            out = []
            for line in candidate_lines:
                if not line_possible(line, ft):
                    continue
                prob = prob_fn(sl, line)
                if 0.05 <= prob <= 0.95:
                    true_o = 1 / prob
                    tgt_o  = true_o * (1 + h_edge)
                    sign   = "+" if line > 0 else ""
                    out.append((f"{sign}{line}", prob, true_o, tgt_o, int(prob * 100)))
            return out

        lines_a = build_lines(handicap_prob_p1)
        lines_b = build_lines(handicap_prob_p2)

        col_ha, col_hb = st.columns(2)

        with col_ha:
            st.markdown(_hcap_card_html(pa, "player-name-a", "card-a", "#4fc3f7", lines_a), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for label, prob, true_o, tgt_o, _ in lines_a:
                safe = label.replace(".", "_").replace("+", "p").replace("-", "m")
                st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-top:6px;'>{pa} {label}</div>", unsafe_allow_html=True)
                mp_val, is_val = market_input_row(f"ha_{safe}", tgt_o, pa, pb, "Handicap", f"{pa} {label}", label)
                if mp_val is not None:
                    st.markdown(value_badge(mp_val, tgt_o, is_val), unsafe_allow_html=True)

        with col_hb:
            st.markdown(_hcap_card_html(pb, "player-name-b", "card-b", "#ef9a9a", lines_b), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for label, prob, true_o, tgt_o, _ in lines_b:
                safe = label.replace(".", "_").replace("+", "p").replace("-", "m")
                st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-top:6px;'>{pb} {label}</div>", unsafe_allow_html=True)
                mp_val, is_val = market_input_row(f"hb_{safe}", tgt_o, pa, pb, "Handicap", f"{pb} {label}", label)
                if mp_val is not None:
                    st.markdown(value_badge(mp_val, tgt_o, is_val), unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:16px;font-family:monospace;font-size:10px;color:#444455;text-align:center;'>"
            "handicap on each player's frame score &nbsp;·&nbsp; lines &lt;5% or &gt;95% hidden</div>",
            unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB 3 — Centuries
# ────────────────────────────────────────────────────────────────────

def _ou_html(color, ou, line, cen_edge):
    ot = 1 / ou["over"]  if ou["over"]  > 0 else float("inf")
    ut = 1 / ou["under"] if ou["under"] > 0 else float("inf")
    otgt = ot * (1 + cen_edge)
    utgt = ut * (1 + cen_edge)
    return (
        f"<div style='margin-bottom:6px;'>"
        f"<div style='font-size:10px;letter-spacing:2px;color:#444455;font-family:monospace;margin-bottom:6px;'>LINE &nbsp;{line}</div>"
        f"<div style='display:flex;gap:8px;'>"
        f"<div class='ou-card' style='flex:1;border-left:3px solid {color};'>"
        f"<div class='ou-label'>OVER {line}</div>"
        f"<div class='ou-prob' style='color:{color};'>{ou['over']*100:.1f}%</div>"
        f"<div style='font-size:20px;font-weight:700;color:{color};margin:4px 0 2px;'>{otgt:.3f}</div>"
        f"<div class='ou-odds'>true {ot:.3f}</div>"
        f"</div>"
        f"<div class='ou-card' style='flex:1;border-left:3px solid #444455;'>"
        f"<div class='ou-label'>UNDER {line}</div>"
        f"<div class='ou-prob' style='color:#888888;'>{ou['under']*100:.1f}%</div>"
        f"<div style='font-size:20px;font-weight:700;color:#888888;margin:4px 0 2px;'>{utgt:.3f}</div>"
        f"<div class='ou-odds'>true {ut:.3f}</div>"
        f"</div></div></div>"
    )


def _dist_html(dist, color, cen_edge):
    collapsed = collapse_dist(dist, threshold=0.02)
    rows = ""
    for label, v in collapsed:
        bw = min(int(v * 200), 100)
        to = 1/v if v > 0 else float("inf")
        tgt = to * (1 + cen_edge)
        rows += (
            f"<tr><td>{label}</td>"
            f"<td><div style='background:#12121e;border-radius:3px;height:8px;width:100%;'>"
            f"<div style='background:{color};border-radius:3px;height:8px;width:{bw}%;'></div></div></td>"
            f"<td>{v*100:.2f}%</td>"
            f"<td style='color:#888888;'>{to:.2f}</td>"
            f"<td style='color:{color};font-weight:700;'>{tgt:.2f}</td>"
            f"</tr>"
        )
    return (
        "<table class='cen-table'><thead><tr>"
        "<th>CENTURIES</th><th style='width:35%;'>DIST</th><th>PROB</th><th>TRUE</th><th>TARGET</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _cen_col(col, name, name_class, card_class, color, ou, line, dist, cen_edge, cr, pa, pb, pkey):
    with col:
        cr_html = (
            f"<div class='breakdown-row' style='margin-bottom:8px;'>"
            f"<span>century rate (given frame win)</span>"
            f"<span class='breakdown-val'>{cr*100:.1f}%</span></div>"
        ) if cr > 0 else ""
        st.markdown(
            f"<div class='{card_class}'><div class='{name_class}'>{name}</div>"
            + cr_html
            + f"<hr class='divider'>"
            + _ou_html(color, ou, line, cen_edge)
            + "<hr class='divider'><div class='section-cap'>FULL DISTRIBUTION</div>"
            + _dist_html(dist, color, cen_edge)
            + "</div>",
            unsafe_allow_html=True
        )
        ot = (1/ou["over"]) * (1+cen_edge) if ou["over"] > 0 else float("inf")
        ut = (1/ou["under"]) * (1+cen_edge) if ou["under"] > 0 else float("inf")
        st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-top:8px;'>OVER {line}</div>", unsafe_allow_html=True)
        mv, iv = market_input_row(f"{pkey}_o", ot, pa, pb, "Centuries", f"{name} Over {line}", line)
        if mv is not None:
            st.markdown(value_badge(mv, ot, iv), unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-top:4px;'>UNDER {line}</div>", unsafe_allow_html=True)
        mv, iv = market_input_row(f"{pkey}_u", ut, pa, pb, "Centuries", f"{name} Under {line}", line)
        if mv is not None:
            st.markdown(value_badge(mv, ut, iv), unsafe_allow_html=True)


with tab_centuries:
    if "result" not in st.session_state:
        st.markdown('<div class="match-info">Select players and press RUN PREDICTION</div>', unsafe_allow_html=True)
    else:
        r = st.session_state["result"]
        pa, pb = st.session_state["players"]
        ft = st.session_state["first_to"]
        cen_edge_pct = st.slider("CENTURIES EDGE TARGET", 0.0, 20.0, 5.0, step=0.5, format="%.1f%%", key="cen_edge")
        cen_edge = cen_edge_pct / 100

        line_options = [x / 2 for x in range(1, ft * 2 + 1, 2)]
        st.markdown("<br>", unsafe_allow_html=True)
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            line_a = line_options[0] if len(line_options) == 1 else st.select_slider(
                f"{pa} LINE", options=line_options, value=line_options[min(1, len(line_options)-1)], key="line_a")
        with lc2:
            line_b = line_options[0] if len(line_options) == 1 else st.select_slider(
                f"{pb} LINE", options=line_options, value=line_options[min(1, len(line_options)-1)], key="line_b")
        with lc3:
            mlo = [x / 2 for x in range(1, ft * 4, 2)]
            line_m = mlo[0] if len(mlo) == 1 else st.select_slider(
                "MATCH LINE", options=mlo, value=mlo[min(2, len(mlo)-1)], key="line_m")

        st.markdown("<br>", unsafe_allow_html=True)
        cen_result = predict_match_centuries(pa, pb, ft, player_rates, r["pfa_blended"])
        ou_a = over_under(cen_result, pa, line_a)
        ou_b = over_under(cen_result, pb, line_b)
        ou_m = over_under(cen_result, "match", line_m)

        col_ca, col_cb, col_cm = st.columns(3)
        cen_rate_a = player_rates.get(pa, 0.0) or 0.0
        cen_rate_b = player_rates.get(pb, 0.0) or 0.0

        _cen_col(col_ca, pa, "player-name-a", "card-a", "#4fc3f7", ou_a, line_a, cen_result[pa], cen_edge, cen_rate_a, pa, pb, "ca")
        _cen_col(col_cb, pb, "player-name-b", "card-b", "#ef9a9a", ou_b, line_b, cen_result[pb], cen_edge, cen_rate_b, pa, pb, "cb")
        _cen_col(col_cm, "MATCH TOTAL", "player-name-match", "card-match", "#b39ddb", ou_m, line_m, cen_result["match"], cen_edge, 0.0, pa, pb, "cm")


# ────────────────────────────────────────────────────────────────────
# TAB 4 — Bet Slip
# ────────────────────────────────────────────────────────────────────

with tab_betslip:
    st.markdown("### 📋 Bet Slip")
    st.markdown("<br>", unsafe_allow_html=True)

    df = st.session_state["bet_slip"]

    if df.empty:
        st.markdown(
            "<div style='text-align:center;color:#444455;font-family:monospace;font-size:13px;padding:32px 0;'>"
            "No bets added yet. Use the ＋ buttons in each tab.</div>",
            unsafe_allow_html=True)
    else:
        # Checkbox column for row deletion
        st.markdown("<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:4px;'>Select rows to delete</div>", unsafe_allow_html=True)

        to_delete = []
        header = (
            "<table class='bet-table'><thead><tr>"
            "<th></th><th>PLAYER 1</th><th>PLAYER 2</th><th>MARKET</th>"
            "<th>SELECTION</th><th>LINE</th><th>TARGET</th><th>MARKET PRICE</th><th>KELLY %</th>"
            "</tr></thead><tbody>"
        )

        # We render the table HTML separately from checkboxes (Streamlit limitation).
        # Show table first, then checkboxes below mapped by index.
        rows_html = ""
        for i, row in df.iterrows():
            mp = float(row["Market Price"])
            tgt = float(row["Target"])
            mp_color = "#a5d6a7" if mp > tgt else "#ef9a9a"
            kelly = row.get("Kelly Stake %", "")
            rows_html += (
                f"<tr>"
                f"<td>{i+1}</td>"
                f"<td>{row['Player 1']}</td><td>{row['Player 2']}</td>"
                f"<td>{row['Market']}</td><td>{row['Selection']}</td>"
                f"<td>{row['Line']}</td>"
                f"<td style='color:#4fc3f7;font-weight:700;'>{tgt:.3f}</td>"
                f"<td style='color:{mp_color};font-weight:700;'>{mp:.3f}</td>"
                f"<td style='color:#b39ddb;font-weight:700;'>{kelly}</td>"
                f"</tr>"
            )
        st.markdown(header + rows_html + "</tbody></table>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;color:#444455;font-family:monospace;'>Select rows to remove:</div>", unsafe_allow_html=True)

        # Render checkboxes in a compact row
        check_cols = st.columns(min(len(df), 12))
        for i, row in df.iterrows():
            col_idx = i % len(check_cols)
            with check_cols[col_idx]:
                if st.checkbox(f"#{i+1}", key=f"del_row_{i}"):
                    to_delete.append(i)

        if to_delete:
            if st.button(f"🗑 Delete selected ({len(to_delete)})", key="del_selected"):
                st.session_state["bet_slip"] = df.drop(index=to_delete).reset_index(drop=True)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    btn1, btn2, _ = st.columns([1, 1, 8])

    with btn1:
        if not st.session_state["bet_slip"].empty:
            csv_data = st.session_state["bet_slip"].to_csv(index=False)
            st.download_button("💾 Save CSV", data=csv_data, file_name="bet_slip.csv",
                               mime="text/csv", key="dl_csv")

    with btn2:
        if st.button("🗑 Clear All", key="clear_bets"):
            st.session_state["bet_slip"] = pd.DataFrame(
                columns=["Player 1", "Player 2", "Market", "Selection", "Line",
                         "Target", "Market Price", "Kelly Stake %"])
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# LOAD MATCH HISTORY
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def load_match_df():
    df = pd.read_csv("player_matches_df.csv")
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df

try:
    match_df = load_match_df()
    _results_available = True
except FileNotFoundError:
    match_df = pd.DataFrame()
    _results_available = False


# ════════════════════════════════════════════════════════════════════
# HELPERS — Results display
# ════════════════════════════════════════════════════════════════════

def _results_table_html(df_in: pd.DataFrame, color: str) -> str:
    """Render a styled results table as HTML. Win rows highlighted faintly."""
    cols = ["match_date", "tournament_name", "round_name", "opposition_name",
            "player_score", "opposition_score", "player_rating", "oppo_rating",
            "player_prob", "player_delta", "player_matches_played", "oppo_matches_played"]
    headers = ["DATE", "TOURNAMENT", "ROUND", "OPPONENT",
               "PTS", "OPP PTS", "RATING", "OPP RTG",
               "WIN PROB", "DELTA", "P MATCHES", "O MATCHES"]

    head_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for _, row in df_in.iterrows():
        won = int(row.get("player_score", 0)) > int(row.get("opposition_score", 0))
        bg  = "background:#1a2a1a;" if won else ""
        delta = row.get("player_delta", 0)
        delta_color = "#a5d6a7" if delta > 0 else "#ef9a9a"
        pm = int(row.get("player_matches_played", 0))
        pm_style = "color:#ef5350;font-weight:700;" if pm < 50 else "color:#cccccc;"
        om = int(row.get("oppo_matches_played", 0))
        om_style = "color:#ef5350;font-weight:700;" if om < 50 else "color:#cccccc;"
        rows_html += (
            f"<tr style='{bg}'>"
            f"<td>{row.get('match_date','')}</td>"
            f"<td style='color:#cccccc;'>{row.get('tournament_name','')}</td>"
            f"<td>{row.get('round_name','')}</td>"
            f"<td style='color:{color};font-weight:700;'>{row.get('opposition_name','')}</td>"
            f"<td style='color:#ffffff;font-weight:700;'>{row.get('player_score','')}</td>"
            f"<td>{row.get('opposition_score','')}</td>"
            f"<td style='color:#4fc3f7;'>{row.get('player_rating', 0):.1f}</td>"
            f"<td style='color:#888888;'>{row.get('oppo_rating', 0):.1f}</td>"
            f"<td>{row.get('player_prob', 0):.1%}</td>"
            f"<td style='color:{delta_color};font-weight:700;'>{delta:+.3f}</td>"
            f"<td style='{pm_style}'>{pm}</td>"
            f"<td style='{om_style}'>{om}</td>"
            f"</tr>"
        )
    return (
        "<table class='bet-table' style='font-size:11px;'>"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _h2h_table_html(df_in: pd.DataFrame, pa: str, color_a: str, color_b: str) -> str:
    headers = ["DATE", "TOURNAMENT", "ROUND", "PLAYER", "PTS", "OPP PTS", "WIN PROB", "DELTA"]
    head_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for _, row in df_in.iterrows():
        is_pa = row["player_name"] == pa
        color = color_a if is_pa else color_b
        won = int(row.get("player_score", 0)) > int(row.get("opposition_score", 0))
        bg  = "background:#1a2a1a;" if won else ""
        delta = row.get("player_delta", 0)
        delta_color = "#a5d6a7" if delta > 0 else "#ef9a9a"
        rows_html += (
            f"<tr style='{bg}'>"
            f"<td>{row.get('match_date','')}</td>"
            f"<td style='color:#cccccc;'>{row.get('tournament_name','')}</td>"
            f"<td>{row.get('round_name','')}</td>"
            f"<td style='color:{color};font-weight:700;'>{row.get('player_name','')}</td>"
            f"<td style='color:#ffffff;font-weight:700;'>{row.get('player_score','')}</td>"
            f"<td>{row.get('opposition_score','')}</td>"
            f"<td>{row.get('player_prob', 0):.1%}</td>"
            f"<td style='color:{delta_color};font-weight:700;'>{delta:+.3f}</td>"
            f"</tr>"
        )
    return (
        "<table class='bet-table' style='font-size:11px;'>"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _rating_chart(df_player: pd.DataFrame, name: str, color: str):
    """Plotly rating-over-time line chart."""
    df_sorted = df_player.sort_values("match_date")
    # Deduplicate by date — take last rating on each date
    df_sorted = df_sorted.drop_duplicates(subset="match_date", keep="last")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["match_date"],
        y=df_sorted["player_rating"],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=4),
        hovertemplate="%{x}<br>Rating: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#13131f",
        plot_bgcolor="#1e1e2e",
        font=dict(family="IBM Plex Mono", color="#888888", size=11),
        margin=dict(l=40, r=20, t=30, b=40),
        height=220,
        xaxis=dict(gridcolor="#2a2a3a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#2a2a3a", showgrid=True, zeroline=False),
        showlegend=False,
        title=dict(text=f"{name} — Rating History", font=dict(color=color, size=13), x=0),
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# TAB 5 — Results
# ════════════════════════════════════════════════════════════════════

with tab_results:
    if not _results_available:
        st.markdown(
            "<div class='match-info'>player_matches_df.csv not found in app directory.</div>",
            unsafe_allow_html=True)
    elif "players" not in st.session_state:
        st.markdown(
            "<div class='match-info'>Run a prediction first to select players.</div>",
            unsafe_allow_html=True)
    else:
        pa, pb = st.session_state["players"]

        # N results slider — outside sub-tabs so it applies to both player views
        n_results = st.slider("Last N results", min_value=5, max_value=100, value=20, step=5, key="n_results")

        sub_pa, sub_pb, sub_h2h = st.tabs([f"👤  {pa}", f"👤  {pb}", "⚔️  H2H"])

        # ── Player A sub-tab ──────────────────────────────────────
        with sub_pa:
            df_a = match_df[match_df["player_name"] == pa].sort_values("match_date", ascending=False)
            if df_a.empty:
                st.markdown(f"<div class='match-info'>No results found for {pa}.</div>", unsafe_allow_html=True)
            else:
                # Rating chart — full history
                st.plotly_chart(_rating_chart(df_a, pa, "#4fc3f7"), use_container_width=True, config={"displayModeBar": False})

                # Recent N results table
                df_a_recent = df_a.head(n_results)
                st.markdown(
                    f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:6px;'>"
                    f"LAST {len(df_a_recent)} RESULTS</div>",
                    unsafe_allow_html=True)
                st.markdown(_results_table_html(df_a_recent, "#ef9a9a"), unsafe_allow_html=True)

        # ── Player B sub-tab ──────────────────────────────────────
        with sub_pb:
            df_b = match_df[match_df["player_name"] == pb].sort_values("match_date", ascending=False)
            if df_b.empty:
                st.markdown(f"<div class='match-info'>No results found for {pb}.</div>", unsafe_allow_html=True)
            else:
                st.plotly_chart(_rating_chart(df_b, pb, "#ef9a9a"), use_container_width=True, config={"displayModeBar": False})

                df_b_recent = df_b.head(n_results)
                st.markdown(
                    f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:6px;'>"
                    f"LAST {len(df_b_recent)} RESULTS</div>",
                    unsafe_allow_html=True)
                st.markdown(_results_table_html(df_b_recent, "#4fc3f7"), unsafe_allow_html=True)

        # ── H2H sub-tab ───────────────────────────────────────────
        with sub_h2h:
            # All rows where pa played pb (from either perspective)
            h2h_df = match_df[
                ((match_df["player_name"] == pa) & (match_df["opposition_name"] == pb)) |
                ((match_df["player_name"] == pb) & (match_df["opposition_name"] == pa))
            ].copy().sort_values("match_date", ascending=False)

            if h2h_df.empty:
                st.markdown(
                    f"<div class='match-info'>No head-to-head results found for {pa} vs {pb}.</div>",
                    unsafe_allow_html=True)
            else:
                # Summary — wins for each player (from their own rows where they won)
                wins_a = ((h2h_df["player_name"] == pa) &
                          (h2h_df["player_score"] > h2h_df["opposition_score"])).sum()
                wins_b = ((h2h_df["player_name"] == pb) &
                          (h2h_df["player_score"] > h2h_df["opposition_score"])).sum()
                total_matches = len(h2h_df) // 2  # two rows per match

                st.markdown(
                    f"<div style='display:flex;gap:24px;align-items:center;margin-bottom:16px;'>"
                    f"<div style='background:#1e1e2e;border:2px solid #4fc3f7;border-radius:8px;padding:14px 24px;text-align:center;'>"
                    f"<div style='font-size:9px;letter-spacing:2px;color:#444455;font-family:monospace;margin-bottom:4px;'>WINS</div>"
                    f"<div style='font-size:36px;font-weight:700;color:#4fc3f7;font-family:Rajdhani,sans-serif;'>{wins_a}</div>"
                    f"<div style='font-size:13px;color:#4fc3f7;font-family:Rajdhani,sans-serif;'>{pa}</div>"
                    f"</div>"
                    f"<div style='font-size:22px;color:#444455;font-family:Rajdhani,sans-serif;font-weight:700;'>vs</div>"
                    f"<div style='background:#1e1e2e;border:2px solid #ef9a9a;border-radius:8px;padding:14px 24px;text-align:center;'>"
                    f"<div style='font-size:9px;letter-spacing:2px;color:#444455;font-family:monospace;margin-bottom:4px;'>WINS</div>"
                    f"<div style='font-size:36px;font-weight:700;color:#ef9a9a;font-family:Rajdhani,sans-serif;'>{wins_b}</div>"
                    f"<div style='font-size:13px;color:#ef9a9a;font-family:Rajdhani,sans-serif;'>{pb}</div>"
                    f"</div>"
                    f"<div style='font-size:11px;color:#444455;font-family:monospace;'>{total_matches} matches</div>"
                    f"</div>",
                    unsafe_allow_html=True)

                # Show last N (both rows per match, but grouped by match_id naturally via sort)
                h2h_recent = h2h_df.head(n_results * 2)
                st.markdown(_h2h_table_html(h2h_recent, pa, "#4fc3f7", "#ef9a9a"), unsafe_allow_html=True)
