import io
import json
import math
import gc
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom
from scipy.special import comb as sp_comb
import plotly.graph_objects as go

# ── Optional deps for the Update tab ─────────────────────────────────
# Scraper (network + parsing). Guarded so the serving app still loads
# even if these aren't installed in some environment.
try:
    import requests
    from bs4 import BeautifulSoup
    import scraper_functions as sf
    _SCRAPER_AVAILABLE = True
    _SCRAPER_ERR = None
except Exception as e:                       # pragma: no cover
    _SCRAPER_AVAILABLE = False
    _SCRAPER_ERR = repr(e)

# GitHub commit layer
try:
    from github import Github, InputGitTreeElement
    _GITHUB_AVAILABLE = True
    _GITHUB_ERR = None
except Exception as e:                       # pragma: no cover
    _GITHUB_AVAILABLE = False
    _GITHUB_ERR = repr(e)


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
    .card-upd { background-color: #1e1e2e; border: 2px solid #2a2a3a; border-radius: 10px; padding: 20px; }
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
    .rank-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
    .rank-table th { color: #444455; font-size: 9px; letter-spacing: 2px; padding: 10px 16px; text-align: left; border-bottom: 2px solid #2a2a3a; position: sticky; top: 0; background: #13131f; }
    .rank-table th.num { text-align: right; }
    .rank-table td { padding: 9px 16px; border-bottom: 1px solid #1a1a2a; vertical-align: middle; }
    .rank-table tr:last-child td { border-bottom: none; }
    .rank-table tr:hover td { background-color: #1a1a2a; }
    .upd-step { font-size: 11px; letter-spacing: 2px; color: #4fc3f7; font-family: 'IBM Plex Mono', monospace; margin: 4px 0; }
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# FILE / REPO CONSTANTS
# ════════════════════════════════════════════════════════════════════

PARQUET_PATH      = "match_frame_player_data.parq"   # master frame-level dataset
RATINGS_PATH      = "ratings.json"                   # {"elob": ..., "elof": ...}
PLAYER_RATES_PATH = "player_rates.json"              # century rates
MATCHES_CSV_PATH  = "player_matches_df.csv"          # historical results + preds
OUTRIGHT_BRACKETS_PATH = "outright_brackets.json"    # saved Outright tab bracket presets

# Columns used to preview newly-scraped matches
MATCH_COLS = ["match_date", "tournament_name", "player_name",
              "opposition_name", "player_score", "opposition_score"]


# ════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════

if "bet_slip" not in st.session_state:
    st.session_state["bet_slip"] = pd.DataFrame(
        columns=["Player 1", "Player 2", "Market", "Selection", "Line",
                 "Target", "Market Price", "Kelly Stake %"]
    )
if "mp_version" not in st.session_state:
    st.session_state["mp_version"] = 0


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def kelly_stake(market_odds: float, prob_win: float) -> float:
    b = market_odds - 1
    q = 1 - prob_win
    if b <= 0:
        return 0.0
    return max((b * prob_win - q) / b, 0.0)


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
    items = sorted(dist.items())
    result, tail_prob, tail_start = [], 0.0, None
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
# MODEL FUNCTIONS — Match prediction (serving)
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
# MODEL FUNCTIONS — Century prediction (serving)
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
# MODEL FUNCTIONS — Outright simulation (serving)
# ════════════════════════════════════════════════════════════════════
# Vectorised (numpy) Monte Carlo bracket simulation. Ratings update
# match-by-match within each simulated tournament using the same
# delta formulas as run_elo_beta / run_elo_frames, so a player's
# projected strength genuinely evolves as the hypothetical draw plays
# out. Chunked over simulations so peak memory stays bounded no matter
# how large N_sims is — only one chunk's arrays are ever live at once.

OUTRIGHT_ELOB_K = 14          # mirrors run_elo_beta's default k
OUTRIGHT_ELOB_K_DECAY = 0.02  # mirrors run_elo_beta's default k_decay
OUTRIGHT_ELOF_K = 16          # mirrors run_elo_frames's default k
OUTRIGHT_CHUNK_SIZE = 4000    # sims processed per batch (bounds peak memory)


def outright_round_plan(x_players: int) -> list:
    """Round names in PLAY order (earliest round first, Final last)."""
    n_rounds = int(round(math.log2(x_players)))
    names = []
    for r in range(n_rounds):
        dist = n_rounds - 1 - r
        if dist == 0:
            names.append("Final")
        elif dist == 1:
            names.append("Semi Final")
        elif dist == 2:
            names.append("Quarter Final")
        else:
            names.append(f"Round of {2 ** (dist + 1)}")
    return names


def _sample_loser_frames(p_win_frame: np.ndarray, first_to: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised draw of the losing side's frame count (0..first_to-1), from the
    scoreline distribution conditional on this side having won the match.
    Same combinatorics as scoreline_probs, worked in log-space for stability
    at large first_to.
    """
    flat = p_win_frame.ravel()
    js = np.arange(first_to)
    log_p = np.log(np.clip(flat, 1e-12, 1.0))
    log_q = np.log(np.clip(1.0 - flat, 1e-12, 1.0))
    log_combs = np.log(sp_comb(first_to - 1 + js, js))
    log_w = log_combs[None, :] + first_to * log_p[:, None] + js[None, :] * log_q[:, None]
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    cdf = np.cumsum(w, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random((flat.shape[0], 1))
    j_idx = (cdf >= u).argmax(axis=1)
    return js[j_idx].reshape(p_win_frame.shape)


def _play_round_vec(idx_a, idx_b, rat_b, rat_f, n_b, n_f, first_to, elob_w, rng):
    """
    Plays one round for every (sim, match) pair at once. idx_a/idx_b are (n_chunk, n_matches)
    roster-index arrays for the two sides. rat_b/rat_f/n_b/n_f are (n_chunk, roster_size) state
    arrays and are updated IN PLACE (winner and loser both get a rating delta + matches_played+1,
    mirroring run_elo_beta/run_elo_frames). Returns the (n_chunk, n_matches) roster-index array
    of winners.
    """
    best_of = 2 * first_to - 1

    ra_b = np.take_along_axis(rat_b, idx_a, axis=1)
    rb_b = np.take_along_axis(rat_b, idx_b, axis=1)
    ra_f = np.take_along_axis(rat_f, idx_a, axis=1)
    rb_f = np.take_along_axis(rat_f, idx_b, axis=1)
    na_b = np.take_along_axis(n_b, idx_a, axis=1)
    nb_b = np.take_along_axis(n_b, idx_b, axis=1)

    pf_b = elo_expected_frame(ra_b, rb_b)
    pf_f = elo_expected_frame(ra_f, rb_f)
    pm_b = p_win_match(pf_b, best_of)
    pm_f = p_win_match(pf_f, best_of)

    prob_a = elob_w * pm_b + (1 - elob_w) * pm_f          # blended match-win prob (matches Match Odds tab)
    pf_blended = elob_w * pf_b + (1 - elob_w) * pf_f      # blended frame prob (drives scoreline sampling,
                                                            # same convention as tab_handicap/tab_centuries)

    a_wins = rng.random(prob_a.shape) < prob_a
    p_win_frame = np.where(a_wins, pf_blended, 1 - pf_blended)
    loser_frames = _sample_loser_frames(p_win_frame, first_to, rng).astype(np.float64)

    a_frames = np.where(a_wins, float(first_to), loser_frames)
    b_frames = np.where(a_wins, loser_frames, float(first_to))
    total_frames = a_frames + b_frames
    ratio_a = a_frames / total_frames
    ratio_b = b_frames / total_frames

    # ELO-beta update (match-outcome model, margin-of-victory + K-decay) — mirrors run_elo_beta
    winner_rat_b = np.where(a_wins, ra_b, rb_b)
    loser_rat_b = np.where(a_wins, rb_b, ra_b)
    mov = np.log(np.abs(a_frames - b_frames) + 1) * (2.2 / ((winner_rat_b - loser_rat_b) * 0.001 + 2.2))
    outcome_a = a_wins.astype(np.float64)
    k1_b = OUTRIGHT_ELOB_K / (1 + OUTRIGHT_ELOB_K_DECAY * na_b)
    k2_b = OUTRIGHT_ELOB_K / (1 + OUTRIGHT_ELOB_K_DECAY * nb_b)
    delta_a_b = k1_b * mov * (outcome_a - pm_b)
    delta_b_b = k2_b * mov * ((1 - outcome_a) - (1 - pm_b))

    # ELO-frames update (frame-ratio model) — mirrors run_elo_frames
    delta_a_f = OUTRIGHT_ELOF_K * (ratio_a - pf_f)
    delta_b_f = OUTRIGHT_ELOF_K * (ratio_b - (1 - pf_f))

    np.put_along_axis(rat_b, idx_a, ra_b + delta_a_b, axis=1)
    np.put_along_axis(rat_b, idx_b, rb_b + delta_b_b, axis=1)
    np.put_along_axis(rat_f, idx_a, ra_f + delta_a_f, axis=1)
    np.put_along_axis(rat_f, idx_b, rb_f + delta_b_f, axis=1)
    np.put_along_axis(n_b, idx_a, na_b + 1, axis=1)
    np.put_along_axis(n_b, idx_b, nb_b + 1, axis=1)

    return np.where(a_wins, idx_a, idx_b)


def _simulate_outright_chunk(roster, x_players, qualifiers, first_to_by_round, elob_w,
                              ratings_elob, ratings_elof, n_chunk, rng):
    """
    Runs n_chunk simulated tournaments. `roster` is the full player-name list: the first
    x_players entries are the Round 1 slot occupants (one per slot — for a slot still pending
    a qualifier, this is that qualifier's "Player A"), followed by one extra entry per pending
    qualifier for its "Player B". `qualifiers` is a list of {"slot", "other_idx", "first_to"}
    dicts — before Round 1, each pending slot's qualifier is resolved (same rating-update
    machinery as any other round) and the winner (with their post-qualifier rating) becomes
    that slot's Round 1 occupant. Returns (champion_counts, finalist_counts), each length
    len(roster).
    """
    r_size = len(roster)

    init_rat_b = np.array([ratings_elob.get(p, {}).get("rating", 0.0) for p in roster], dtype=np.float64)
    init_rat_f = np.array([ratings_elof.get(p, {}).get("rating", 0.0) for p in roster], dtype=np.float64)
    init_n_b = np.array([ratings_elob.get(p, {}).get("matches_played", 0) for p in roster], dtype=np.float64)
    init_n_f = np.array([ratings_elof.get(p, {}).get("matches_played", 0) for p in roster], dtype=np.float64)

    rat_b = np.tile(init_rat_b, (n_chunk, 1))
    rat_f = np.tile(init_rat_f, (n_chunk, 1))
    n_b = np.tile(init_n_b, (n_chunk, 1))
    n_f = np.tile(init_n_f, (n_chunk, 1))

    occupant = np.tile(np.arange(x_players, dtype=np.int32), (n_chunk, 1))

    if qualifiers:
        slot_idx = np.array([q["slot"] for q in qualifiers], dtype=np.int32)
        other_idx = np.array([q["other_idx"] for q in qualifiers], dtype=np.int32)
        qual_ft = qualifiers[0]["first_to"]   # one shared qualifying format for this bracket
        idx_a = occupant[:, slot_idx]
        idx_b = np.tile(other_idx, (n_chunk, 1))
        winners = _play_round_vec(idx_a, idx_b, rat_b, rat_f, n_b, n_f, qual_ft, elob_w, rng)
        occupant[:, slot_idx] = winners

    finalists = None
    for first_to in first_to_by_round:
        if occupant.shape[1] == 2:
            finalists = occupant.copy()
        idx_a = occupant[:, 0::2]
        idx_b = occupant[:, 1::2]
        occupant = _play_round_vec(idx_a, idx_b, rat_b, rat_f, n_b, n_f, first_to, elob_w, rng)

    champion_counts = np.bincount(occupant.ravel(), minlength=r_size)
    finalist_counts = np.bincount(finalists.ravel(), minlength=r_size)
    return champion_counts, finalist_counts


def run_outright_simulation(roster, x_players, qualifiers, first_to_by_round, n_sims, elob_w,
                            ratings_elob, ratings_elof, progress_cb=None, seed=None):
    """Chunked orchestrator. Returns {"champion_prob": {...}, "finalist_prob": {...}, "n_sims": n}."""
    rng = np.random.default_rng(seed)
    r_size = len(roster)
    champion_counts = np.zeros(r_size, dtype=np.int64)
    finalist_counts = np.zeros(r_size, dtype=np.int64)

    done = 0
    while done < n_sims:
        n_chunk = min(OUTRIGHT_CHUNK_SIZE, n_sims - done)
        champ, fin = _simulate_outright_chunk(
            roster, x_players, qualifiers, first_to_by_round,
            elob_w, ratings_elob, ratings_elof, n_chunk, rng)
        champion_counts += champ
        finalist_counts += fin
        done += n_chunk
        if progress_cb:
            progress_cb(done, n_sims)

    return {
        "champion_prob": {p: champion_counts[i] / n_sims for i, p in enumerate(roster)},
        "finalist_prob": {p: finalist_counts[i] / n_sims for i, p in enumerate(roster)},
        "n_sims": n_sims,
    }


# ════════════════════════════════════════════════════════════════════
# DATA PIPELINE — scraping  (mirrors update_data.ipynb)
# ════════════════════════════════════════════════════════════════════
# NOTE: these orchestration functions are reproduced here from your
# update_data notebook, because scraper_functions.py only exposes the
# low-level parsers (get_frame_level_data etc.). The low-level parsing
# is still delegated to scraper_functions.get_frame_level_data.

def get_tournament_urls(seasons: list):
    url_base = (lambda season: f"https://cuetracker.net/seasons/{str(season-1)}-{str(season)}")
    tourn_links = []
    for season in seasons:
        response = requests.get(url_base(season))
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", class_="table-striped")
        _tmp = [a["href"] for a in table.find("tbody").find_all("a") if "tournaments" in a["href"]]
        tourn_links.extend(_tmp)
    return tourn_links


def get_tournament_info(soup):
    name = soup.find("h1", class_="text-center").text.strip()
    info_table = soup.find("table", class_="table-small")
    status = None
    location = None
    for row in info_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            label = cells[0].text.strip()
            if "Status" in label:
                status = cells[1].text.strip()
            elif "Location" in label:
                location = ", ".join(a.text.strip() for a in cells[1].find_all("a"))
    return {"name": name, "status": status, "location": location}


def get_tournament_matches(tourn_links, progress_cb=None):
    full_data = []
    for tourn in tourn_links:
        if progress_cb:
            progress_cb(f"  · {tourn}")
        try:
            response = requests.get(tourn)
            soup = BeautifulSoup(response.content, "html.parser")
            tourn_data = get_tournament_info(soup)
            data = sf.get_frame_level_data(soup)
        except ValueError:
            # No matches on this page yet (e.g. an upcoming tournament) ->
            # get_frame_level_data hits pd.concat([]) and raises. Skip it.
            if progress_cb:
                progress_cb("    skipped (no matches found)")
            continue
        if data is None or len(data) == 0 or "match_date" not in data.columns:
            if progress_cb:
                progress_cb("    skipped (no match_date)")
            continue
        data["tournament_name"] = tourn_data.get("name")
        data["status"] = tourn_data.get("status")
        data["location"] = tourn_data.get("location")
        full_data.append(data)
    if not full_data:
        raise RuntimeError("No completed matches found for the requested seasons.")
    full_data = pd.concat(full_data)
    full_data[["match_id", "frame_number"]] = full_data[["match_id", "frame_number"]].astype(int)
    return full_data.sort_values(["match_date", "match_id", "frame_number"]).reset_index(drop=True)


def _normalize_names(df, cols=("player_name", "opposition_name",
                               "player_1_name", "player_2_name")):
    """
    Make player-name strings consistent so freshly-scraped rows share the
    exact same key as the historical data. Fixes hidden whitespace /
    non-breaking spaces that otherwise split one player into two keys and
    break the 2-rows-per-match assumption in the ELO loop.
    """
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("\u00a0", " ", regex=False)   # non-breaking space
                .str.replace(r"\s+", " ", regex=True)        # collapse whitespace
                .str.strip()
            )
    return df


def _add_extra_details(df):
    df = _normalize_names(df)
    cols = ["player_1_score", "player_2_score", "best_of",
            "player_1_frame_score", "player_2_frame_score"]
    df[cols] = df[cols].astype(float)
    df["player_1_match_win"] = np.select(
        [df.player_1_score > df.player_2_score, df.player_1_score < df.player_2_score,
         df.player_1_score == df.player_2_score], [1, 0, 0.5], np.nan)
    df["player_2_match_win"] = np.select(
        [df.player_1_score > df.player_2_score, df.player_1_score < df.player_2_score,
         df.player_1_score == df.player_2_score], [0, 1, 0.5], np.nan)
    df["player_1_frame_win"] = df.player_1_frame_score > df.player_2_frame_score
    df["player_2_frame_win"] = df.player_2_frame_score > df.player_1_frame_score
    df["player_1_cen"] = df.player_1_50_break >= 100
    df["player_2_cen"] = df.player_2_50_break >= 100
    df["player_1_pre_frame_score"] = df.groupby("match_id").player_1_frame_win.transform("cumsum") - df.player_1_frame_win
    df["player_2_pre_frame_score"] = df.groupby("match_id").player_2_frame_win.transform("cumsum") - df.player_2_frame_win
    return df


def match_frame_to_player_level(df):
    p1_cols = [c for c in df.columns if c.startswith("player_1_")]
    p2_cols = [c for c in df.columns if c.startswith("player_2_")]
    context_cols = [c for c in df.columns if c not in p1_cols + p2_cols]

    def build_perspective(df, player_prefix, opposition_prefix):
        player_cols = [c for c in df.columns if c.startswith(player_prefix)]
        oppo_cols   = [c for c in df.columns if c.startswith(opposition_prefix)]
        out = df[context_cols].copy()
        for col in player_cols:
            out[col.replace(player_prefix, "player_")] = df[col]
        for col in oppo_cols:
            out[col.replace(opposition_prefix, "opposition_")] = df[col]
        return out

    p1 = build_perspective(df, "player_1_", "player_2_")
    p2 = build_perspective(df, "player_2_", "player_1_")
    return (pd.concat([p1, p2])
            .sort_values(["match_id", "frame_number"])
            .reset_index(drop=True))


def load_all_snooker_data(seasons, progress_cb=None):
    """
    Prefer the real scraper_functions.load_all_snooker_data if your module
    defines it; otherwise rebuild it from the notebook orchestration above.
    """
    if _SCRAPER_AVAILABLE and hasattr(sf, "load_all_snooker_data"):
        try:
            return sf.load_all_snooker_data(seasons)
        except TypeError:
            return sf.load_all_snooker_data(seasons=seasons)
    if progress_cb:
        progress_cb("Fetching tournament URLs…")
    links = get_tournament_urls(seasons)
    if progress_cb:
        progress_cb(f"{len(links)} tournaments found. Scraping match pages…")
    data = get_tournament_matches(links, progress_cb=progress_cb)
    data = _add_extra_details(data)
    return data


def scrape_and_diff(seasons, min_date="2026-01-01", progress_cb=None):
    """
    Scrape, merge with master parquet, return (combined_df, new_match_ids, preview_df).

    min_date ("YYYY-MM-DD"): only matches played on or after this date are
    considered when detecting which games are new. match_date is stored as an
    ISO-prefixed string, so a lexicographic compare against an ISO date works.
    """
    def log(m):
        if progress_cb:
            progress_cb(m)

    data = load_all_snooker_data(seasons, progress_cb=progress_cb)
    log(f"Scraped {len(data):,} frame rows. Reshaping to player level…")
    player_df = match_frame_to_player_level(data)
    player_df = _normalize_names(player_df)

    log("Loading existing master dataset…")
    output = pd.read_parquet(PARQUET_PATH)
    output = _normalize_names(output)
    existing_matches = output[output.match_date >= min_date].match_id.unique()

    combined = (
        pd.concat([output, player_df])
        .drop_duplicates(["match_id", "player_name", "frame_number"], keep="last")
    )
    output_matches = combined[combined.match_date >= min_date].match_id.unique()
    new_matches = [m for m in output_matches if m not in existing_matches]

    preview = (
        combined[combined.match_id.isin(new_matches)][["match_id"] + MATCH_COLS]
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )
    log(f"{len(new_matches)} new match(es) on or after {min_date}.")
    return combined, new_matches, preview


# ════════════════════════════════════════════════════════════════════
# DATA PIPELINE — ELO models  (mirrors match_models_update.ipynb)
# ════════════════════════════════════════════════════════════════════

def _prepare_frame_df(df):
    df = _normalize_names(df)
    # After normalising names, collapse any rows that became duplicates because
    # the same frame was previously stored under two name spellings.
    df = df.drop_duplicates(["match_id", "player_name", "frame_number"], keep="last")
    df = df[
        ~df.tournament_name.astype(str).str.contains("Shoot")
        & ~df.tournament_name.astype(str).str.contains("6-Reds")
        & (df.match_date.astype(str) != "0000-00-00")
    ].copy()
    df["match_date"] = pd.to_datetime(df["match_date"].astype(str).str.split(" - ").str[0].str.strip())
    df["best_of"] = df["best_of"].astype(int)
    return df


def mov_multiplier(frames_won, frames_lost, elo_diff):
    margin = frames_won - frames_lost
    log_margin = math.log(margin + 1)
    autocorr = 2.2 / (elo_diff * 0.001 + 2.2)
    return log_margin * autocorr


def calculate_k_weight(tournament_name: str):
    if "Championship League" in tournament_name:
        return 0.75
    elif "Q School" in tournament_name:
        return 0.25
    return 1


def run_elo_beta(df, k=14, initial_rating=0.01, k_decay=0.02):
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    match_cols = [
        "match_id", "match_date", "best_of", "tournament_name",
        "player_name", "opposition_name",
        "player_score", "opposition_score",
        "player_match_win", "opposition_match_win",
    ]
    matches = (
        df[match_cols]
        .drop_duplicates(subset=["match_id", "player_name"])
        .sort_values("match_date")
        .reset_index(drop=True)
    )
    ratings, match_counts, records = {}, {}, []

    for match_id, grp in matches.groupby("match_id", sort=False):
        if len(grp) != 2:
            continue
        row1, row2 = grp.iloc[0], grp.iloc[1]
        p1, p2 = row1["player_name"], row2["player_name"]
        r1 = ratings.get(p1, initial_rating)
        r2 = ratings.get(p2, initial_rating)
        n1 = match_counts.get(p1, 0)
        n2 = match_counts.get(p2, 0)

        tourn_k_weight = calculate_k_weight(tournament_name=row1["tournament_name"])
        k1 = k * (1 / (1 + k_decay * n1)) * tourn_k_weight
        k2 = k * (1 / (1 + k_decay * n2)) * tourn_k_weight

        match_counts[p1] = n1 + 1
        match_counts[p2] = n2 + 1

        p_frame_1 = elo_expected_frame(r1, r2)
        p_frame_2 = 1 - p_frame_1

        best_of = row1["best_of"]
        prob1 = p_win_match(p_frame_1, best_of)
        prob2 = 1 - prob1

        frames_won_1 = float(row1["player_score"])
        frames_won_2 = float(row2["player_score"])
        outcome1 = float(row1["player_match_win"])
        outcome2 = float(row2["player_match_win"])

        if outcome1 == 1:
            winner_elo_diff = r1 - r2
            mov = mov_multiplier(frames_won_1, frames_won_2, winner_elo_diff)
        else:
            winner_elo_diff = r2 - r1
            mov = mov_multiplier(frames_won_2, frames_won_1, winner_elo_diff)

        delta1 = k1 * mov * (outcome1 - prob1)
        delta2 = k2 * mov * (outcome2 - prob2)

        records.append({
            "match_id": match_id, "player_name": p1,
            "player_rating": round(r1, 2), "oppo_rating": round(r2, 2),
            "player_prob": round(prob1, 4), "oppo_prob": round(prob2, 4),
            "player_delta": round(delta1, 4), "oppo_delta": round(delta2, 4),
            "player_k": round(k1, 4), "oppo_k": round(k2, 4),
            "player_mov": round(mov, 4),
            "player_matches_played": n1, "oppo_matches_played": n2,
        })
        records.append({
            "match_id": match_id, "player_name": p2,
            "player_rating": round(r2, 2), "oppo_rating": round(r1, 2),
            "player_prob": round(prob2, 4), "oppo_prob": round(prob1, 4),
            "player_delta": round(delta2, 4), "oppo_delta": round(delta1, 4),
            "player_k": round(k2, 4), "oppo_k": round(k1, 4),
            "player_mov": round(mov, 4),
            "player_matches_played": n2, "oppo_matches_played": n1,
        })
        ratings[p1] = r1 + delta1
        ratings[p2] = r2 + delta2

    elo_df = pd.DataFrame(records)
    df_out = df.merge(
        elo_df[["match_id", "player_name",
                "player_rating", "oppo_rating",
                "player_prob", "oppo_prob",
                "player_delta", "oppo_delta",
                "player_k", "oppo_k", "player_mov",
                "player_matches_played", "oppo_matches_played"]],
        on=["match_id", "player_name"], how="left",
    )
    final_ratings = {
        player: {"rating": round(ratings[player], 2),
                 "matches_played": match_counts.get(player, 0)}
        for player in ratings
    }
    return df_out, final_ratings


def run_elo_frames(df, k=16, initial_rating=0):
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    match_cols = [
        "match_id", "match_date", "best_of", "tournament_name",
        "player_name", "opposition_name",
        "player_score", "opposition_score",
        "player_match_win", "opposition_match_win",
    ]
    matches = (
        df[match_cols]
        .drop_duplicates(subset=["match_id", "player_name"])
        .sort_values("match_date")
        .reset_index(drop=True)
    )
    ratings, match_counts, records = {}, {}, []

    for match_id, grp in matches.groupby("match_id", sort=False):
        if len(grp) != 2:
            continue
        row1, row2 = grp.iloc[0], grp.iloc[1]
        p1, p2 = row1["player_name"], row2["player_name"]
        r1 = ratings.get(p1, initial_rating)
        r2 = ratings.get(p2, initial_rating)

        match_counts[p1] = match_counts.get(p1, 0) + 1
        match_counts[p2] = match_counts.get(p2, 0) + 1

        exp_frame_1 = elo_expected_frame(r1, r2)
        exp_frame_2 = 1 - exp_frame_1

        best_of = row1["best_of"]
        prob1 = p_win_match(exp_frame_1, best_of)
        prob2 = 1 - prob1

        frames_p1 = float(row1["player_score"])
        frames_p2 = float(row2["player_score"])
        total_frames = frames_p1 + frames_p2
        if total_frames > 0:
            ratio1 = frames_p1 / total_frames
            ratio2 = frames_p2 / total_frames
        else:
            ratio1 = ratio2 = 0.5

        tourn_k_weight = calculate_k_weight(tournament_name=row1["tournament_name"])
        delta1 = k * (ratio1 - exp_frame_1) * tourn_k_weight
        delta2 = k * (ratio2 - exp_frame_2) * tourn_k_weight

        records.append({
            "match_id": match_id, "player_name": p1,
            "player_rating": round(r1, 2), "oppo_rating": round(r2, 2),
            "player_prob": round(prob1, 4), "oppo_prob": round(prob2, 4),
            "player_delta": round(delta1, 4), "oppo_delta": round(delta2, 4),
            "player_matches_played": match_counts[p1] - 1,
            "oppo_matches_played": match_counts[p2] - 1,
        })
        records.append({
            "match_id": match_id, "player_name": p2,
            "player_rating": round(r2, 2), "oppo_rating": round(r1, 2),
            "player_prob": round(prob2, 4), "oppo_prob": round(prob1, 4),
            "player_delta": round(delta2, 4), "oppo_delta": round(delta1, 4),
            "player_matches_played": match_counts[p2] - 1,
            "oppo_matches_played": match_counts[p1] - 1,
        })
        ratings[p1] = r1 + delta1
        ratings[p2] = r2 + delta2

    elo_df = pd.DataFrame(records)
    df_out = df.merge(
        elo_df[["match_id", "player_name",
                "player_rating", "oppo_rating",
                "player_prob", "oppo_prob",
                "player_delta", "oppo_delta",
                "player_matches_played", "oppo_matches_played"]],
        on=["match_id", "player_name"], how="left",
    )
    final_ratings = {
        player: {"rating": round(ratings[player], 2),
                 "matches_played": match_counts.get(player, 0)}
        for player in ratings
    }
    return df_out, final_ratings


# ════════════════════════════════════════════════════════════════════
# DATA PIPELINE — Century rates  (mirrors century_functions.ipynb)
# ════════════════════════════════════════════════════════════════════

def add_dw_century_rate(df, halflife=220):
    """
    Downweighted (time-decayed) average century-making probability per player,
    given a frame win, with partial credit for near-centuries.

    Vectorized equivalent of the notebook's per-group calc_dw, written to be
    independent of the pandas groupby.apply behaviour (which differs between
    pandas <2.2 and >=3.0).
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["cen_wf"] = np.select(
        [
            (df.player_frame_score <= df.opposition_frame_score),
            (df.player_frame_score > df.opposition_frame_score) & (df.player_50_break >= 100),
            (df.player_frame_score > df.opposition_frame_score) & (df.player_50_break.between(95, 100, inclusive="left")),
            (df.player_frame_score > df.opposition_frame_score) & (df.player_50_break.between(90, 95, inclusive="left")),
            (df.player_frame_score > df.opposition_frame_score) & (df.player_50_break.between(85, 90, inclusive="left")),
        ],
        [np.nan, 1, 0.20, 0.125, 0.05],
        default=0,
    )
    decay = np.log(2) / halflife

    # Only frames the player won contribute (cen_wf not NaN)
    sub = df.dropna(subset=["cen_wf"]).copy()
    if sub.empty:
        df["dw_cen_rate"] = np.nan
        return df, {p: np.nan for p in df["player_name"].unique()}

    # Days measured from each player's most recent contributing frame
    sub["_max_date"] = sub.groupby("player_name")["match_date"].transform("max")
    sub["_days_ago"] = (sub["_max_date"] - sub["match_date"]).dt.days
    sub["_w"] = np.exp(-decay * sub["_days_ago"])
    sub["_wx"] = sub["cen_wf"] * sub["_w"]

    agg = sub.groupby("player_name").agg(_num=("_wx", "sum"), _den=("_w", "sum"))
    rates = (agg["_num"] / agg["_den"])
    player_rates = rates.to_dict()

    # Players with no contributing frames (all NaN) -> NaN, matching original
    for p in df["player_name"].unique():
        player_rates.setdefault(p, np.nan)

    df["dw_cen_rate"] = df["player_name"].map(player_rates)
    return df, player_rates


# ════════════════════════════════════════════════════════════════════
# DATA PIPELINE — orchestrator + GitHub commit
# ════════════════════════════════════════════════════════════════════

def run_models_pipeline(df_frames, elob_k=14, elof_k=16, halflife=220, progress_cb=None):
    """Run both ELO models + the centuries model. Returns the three artifacts."""
    def log(m):
        if progress_cb:
            progress_cb(m)

    log("Preparing frame-level data…")
    df = _prepare_frame_df(df_frames)

    # Visibility: the ELO loop skips any match without exactly 2 player rows.
    # This is usually caused by a player-name mismatch splitting one side in two.
    _per_match = df.drop_duplicates(["match_id", "player_name"]).groupby("match_id").size()
    _bad = _per_match[_per_match != 2]
    if len(_bad):
        log(f"⚠ {len(_bad)} match(es) don't have exactly 2 players and will be "
            f"skipped by ELO — likely a name mismatch. Example match_ids: "
            f"{list(_bad.index[:5])}")

    log("Running ELO-beta (match-outcome model)…")
    df_elob, final_ratings_elob = run_elo_beta(df.copy(), k=elob_k)

    log("Running ELO-frames model…")
    _df_elof, final_ratings_elof = run_elo_frames(df.copy(), k=elof_k)
    del _df_elof
    gc.collect()

    log("Building results table (player_matches_df)…")
    df_elob["player_century"] = df_elob.player_50_break >= 100
    df_elob["oppo_century"] = df_elob.opposition_50_break >= 100
    df_elob["player_centuries"] = df_elob.groupby(["match_id", "player_name"]).player_century.transform("sum")
    df_elob["oppo_centuries"] = df_elob.groupby(["match_id", "player_name"]).oppo_century.transform("sum")
    matches_df = (
        df_elob[[
            "match_id", "round_name", "best_of", "match_date",
            "tournament_name", "player_name", "player_score",
            "opposition_name", "opposition_score", "player_rating",
            "oppo_rating", "player_prob", "oppo_prob", "player_delta",
            "player_matches_played", "oppo_matches_played",
            "player_centuries", "oppo_centuries",
        ]]
        .drop_duplicates(["match_id", "player_name"])
        .reset_index(drop=True)
    )
    del df_elob
    gc.collect()

    log("Running centuries model…")
    _, player_rates = add_dw_century_rate(df, halflife=halflife)

    ratings_obj = {"elob": final_ratings_elob, "elof": final_ratings_elof}
    log("Models complete.")
    return ratings_obj, player_rates, matches_df


def _gh_config():
    """Returns (token, repo, branch) from st.secrets, or (None, None, None)."""
    try:
        gh = st.secrets["github"]
        return gh["token"], gh["repo"], gh.get("branch", "main")
    except Exception:
        return None, None, None


def commit_files_to_github(token, repo_name, branch, files: dict, message: str):
    """
    Commit multiple files in a single commit via the Git Data API.
    `files` maps repo-relative path -> bytes (binary) or str (text).
    Returns the new commit SHA.
    """
    g = Github(token)
    repo = g.get_repo(repo_name)
    ref = repo.get_git_ref(f"heads/{branch}")
    latest_commit = repo.get_git_commit(ref.object.sha)
    base_tree = latest_commit.tree

    elements = []
    for path, content in files.items():
        if isinstance(content, bytes):
            blob = repo.create_git_blob(base64.b64encode(content).decode(), "base64")
        else:
            blob = repo.create_git_blob(content, "utf-8")
        elements.append(InputGitTreeElement(path=path, mode="100644", type="blob", sha=blob.sha))

    new_tree = repo.create_git_tree(elements, base_tree)
    new_commit = repo.create_git_commit(message, new_tree, [latest_commit])
    ref.edit(new_commit.sha)
    return new_commit.sha


def load_saved_outright_brackets() -> dict:
    """Named Outright bracket presets, keyed by name. Not st.cache_data'd — always read fresh
    so a save in this session is immediately visible without waiting on a redeploy."""
    try:
        with open(OUTRIGHT_BRACKETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_outright_bracket(name: str, definition: dict):
    """Writes the bracket locally (so it's usable this session even offline) and, if GitHub
    is configured, commits it too (so it survives a redeploy / is visible to other sessions)."""
    brackets = load_saved_outright_brackets()
    brackets[name] = definition
    data_bytes = json.dumps(brackets, indent=2).encode("utf-8")

    with open(OUTRIGHT_BRACKETS_PATH, "wb") as f:
        f.write(data_bytes)

    token, repo_name, branch = _gh_config()
    if _GITHUB_AVAILABLE and token:
        msg = f"Save outright bracket '{name}' via app {datetime.utcnow():%Y-%m-%d %H:%M UTC}"
        commit_files_to_github(token, repo_name, branch, {OUTRIGHT_BRACKETS_PATH: data_bytes}, msg)


# ════════════════════════════════════════════════════════════════════
# LOAD DATA (serving)
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def load_ratings():
    with open(RATINGS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["elob"], data["elof"]

@st.cache_data
def load_player_rates():
    with open(PLAYER_RATES_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_match_df():
    df = pd.read_csv(MATCHES_CSV_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df

ratings_elob, ratings_elof = load_ratings()
player_rates = load_player_rates()
sorted_players = sorted(ratings_elob.keys(), key=lambda p: ratings_elob[p]["rating"], reverse=True)

try:
    match_df = load_match_df()
    _results_available = True
except FileNotFoundError:
    match_df = pd.DataFrame()
    _results_available = False


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
    st.session_state["mp_version"] = st.session_state.get("mp_version", 0) + 1


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════

(tab_match, tab_handicap, tab_centuries, tab_multi, tab_outright, tab_betslip,
 tab_results, tab_rankings, tab_update) = st.tabs([
    "📊  MATCH ODDS", "↕️  HANDICAPS", "🔴  CENTURIES", "🎯  MULTI MATCH", "🔱  OUTRIGHT",
    "📋  BET SLIP", "📅  RESULTS", "🏆  RANKINGS", "⚙️  UPDATE",
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
            + cr_html + f"<hr class='divider'>"
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
        to_delete = []
        header = (
            "<table class='bet-table'><thead><tr>"
            "<th></th><th>PLAYER 1</th><th>PLAYER 2</th><th>MARKET</th>"
            "<th>SELECTION</th><th>LINE</th><th>TARGET</th><th>MARKET PRICE</th><th>KELLY %</th>"
            "</tr></thead><tbody>"
        )
        rows_html = ""
        for i, row in df.iterrows():
            mp = float(row["Market Price"])
            tgt = float(row["Target"])
            mp_color = "#a5d6a7" if mp > tgt else "#ef9a9a"
            kelly = row.get("Kelly Stake %", "")
            rows_html += (
                f"<tr><td>{i+1}</td>"
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
        check_cols = st.columns(min(len(df), 12))
        for i, row in df.iterrows():
            with check_cols[i % len(check_cols)]:
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


# ────────────────────────────────────────────────────────────────────
# TAB 5 — Results
# ────────────────────────────────────────────────────────────────────

def _results_table_html(df_in: pd.DataFrame, color: str) -> str:
    headers = ["DATE", "TOURNAMENT", "ROUND", "OPPONENT",
               "PTS", "OPP PTS", "RATING", "OPP RTG",
               "WIN PROB", "DELTA", "P MATCHES", "O MATCHES"]
    head_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for _, row in df_in.iterrows():
        won = float(row.get("player_score", 0)) > float(row.get("opposition_score", 0))
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
        won = float(row.get("player_score", 0)) > float(row.get("opposition_score", 0))
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
    df_sorted = df_player.sort_values("match_date").drop_duplicates(subset="match_date", keep="last")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["match_date"], y=df_sorted["player_rating"],
        mode="lines+markers", name=name,
        line=dict(color=color, width=2), marker=dict(color=color, size=4),
        hovertemplate="%{x}<br>Rating: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#13131f", plot_bgcolor="#1e1e2e",
        font=dict(family="IBM Plex Mono", color="#888888", size=11),
        margin=dict(l=40, r=20, t=30, b=40), height=220,
        xaxis=dict(gridcolor="#2a2a3a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#2a2a3a", showgrid=True, zeroline=False),
        showlegend=False,
        title=dict(text=f"{name} — Rating History", font=dict(color=color, size=13), x=0),
    )
    return fig

with tab_results:
    if not _results_available:
        st.markdown("<div class='match-info'>player_matches_df.csv not found in app directory.</div>", unsafe_allow_html=True)
    elif "players" not in st.session_state:
        st.markdown("<div class='match-info'>Run a prediction first to select players.</div>", unsafe_allow_html=True)
    else:
        pa, pb = st.session_state["players"]
        n_results = st.slider("Last N results", min_value=5, max_value=100, value=20, step=5, key="n_results")
        sub_pa, sub_pb, sub_h2h = st.tabs([f"👤  {pa}", f"👤  {pb}", "⚔️  H2H"])

        with sub_pa:
            df_a = match_df[match_df["player_name"] == pa].sort_values("match_date", ascending=False)
            if df_a.empty:
                st.markdown(f"<div class='match-info'>No results found for {pa}.</div>", unsafe_allow_html=True)
            else:
                st.plotly_chart(_rating_chart(df_a, pa, "#4fc3f7"), use_container_width=True, config={"displayModeBar": False})
                df_a_recent = df_a.head(n_results)
                st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:6px;'>LAST {len(df_a_recent)} RESULTS</div>", unsafe_allow_html=True)
                st.markdown(_results_table_html(df_a_recent, "#ef9a9a"), unsafe_allow_html=True)

        with sub_pb:
            df_b = match_df[match_df["player_name"] == pb].sort_values("match_date", ascending=False)
            if df_b.empty:
                st.markdown(f"<div class='match-info'>No results found for {pb}.</div>", unsafe_allow_html=True)
            else:
                st.plotly_chart(_rating_chart(df_b, pb, "#ef9a9a"), use_container_width=True, config={"displayModeBar": False})
                df_b_recent = df_b.head(n_results)
                st.markdown(f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:6px;'>LAST {len(df_b_recent)} RESULTS</div>", unsafe_allow_html=True)
                st.markdown(_results_table_html(df_b_recent, "#4fc3f7"), unsafe_allow_html=True)

        with sub_h2h:
            h2h_df = match_df[
                ((match_df["player_name"] == pa) & (match_df["opposition_name"] == pb)) |
                ((match_df["player_name"] == pb) & (match_df["opposition_name"] == pa))
            ].copy().sort_values("match_date", ascending=False)

            if h2h_df.empty:
                st.markdown(f"<div class='match-info'>No head-to-head results found for {pa} vs {pb}.</div>", unsafe_allow_html=True)
            else:
                wins_a = ((h2h_df["player_name"] == pa) & (h2h_df["player_score"] > h2h_df["opposition_score"])).sum()
                wins_b = ((h2h_df["player_name"] == pb) & (h2h_df["player_score"] > h2h_df["opposition_score"])).sum()
                total_matches = len(h2h_df) // 2
                st.markdown(
                    f"<div style='display:flex;gap:24px;align-items:center;margin-bottom:16px;'>"
                    f"<div style='background:#1e1e2e;border:2px solid #4fc3f7;border-radius:8px;padding:14px 24px;text-align:center;'>"
                    f"<div style='font-size:9px;letter-spacing:2px;color:#444455;font-family:monospace;margin-bottom:4px;'>WINS</div>"
                    f"<div style='font-size:36px;font-weight:700;color:#4fc3f7;font-family:Rajdhani,sans-serif;'>{wins_a}</div>"
                    f"<div style='font-size:13px;color:#4fc3f7;font-family:Rajdhani,sans-serif;'>{pa}</div></div>"
                    f"<div style='font-size:22px;color:#444455;font-family:Rajdhani,sans-serif;font-weight:700;'>vs</div>"
                    f"<div style='background:#1e1e2e;border:2px solid #ef9a9a;border-radius:8px;padding:14px 24px;text-align:center;'>"
                    f"<div style='font-size:9px;letter-spacing:2px;color:#444455;font-family:monospace;margin-bottom:4px;'>WINS</div>"
                    f"<div style='font-size:36px;font-weight:700;color:#ef9a9a;font-family:Rajdhani,sans-serif;'>{wins_b}</div>"
                    f"<div style='font-size:13px;color:#ef9a9a;font-family:Rajdhani,sans-serif;'>{pb}</div></div>"
                    f"<div style='font-size:11px;color:#444455;font-family:monospace;'>{total_matches} matches</div>"
                    f"</div>",
                    unsafe_allow_html=True)
                st.markdown(_h2h_table_html(h2h_df.head(n_results * 2), pa, "#4fc3f7", "#ef9a9a"), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB — Multi Match  (simple inputs -> target prices for many matches)
# ────────────────────────────────────────────────────────────────────

with tab_multi:
    st.markdown("### 🎯 Multi Match")
    st.markdown(
        "<div style='font-size:11px;color:#888888;font-family:monospace;margin-bottom:12px;'>"
        "Price several matches at once. ELOb weight and Edge below apply to every match."
        "</div>", unsafe_allow_html=True)

    # ---- Top-level controls (apply to all matches) --------------------------
    mtc1, mtc2, mtc3, mtc4 = st.columns([1, 1, 1, 1.2])
    with mtc1:
        mm_n = st.number_input("Number of matches", min_value=1, max_value=30,
                               value=3, step=1, key="mm_n")
    with mtc2:
        mm_elob_w = st.slider("ELOb weight", 0.0, 1.0, 0.8, step=0.05,
                              format="%.2f", key="mm_elob_w")
    with mtc3:
        mm_edge_pct = st.slider("Edge target", 0.0, 20.0, 5.0, step=0.5,
                                format="%.1f%%", key="mm_edge")
    with mtc4:
        mm_century = st.checkbox("Century preds", value=False, key="mm_century",
                                 help="Adds an O/U centuries line per match and outputs "
                                      "over/under target prices.")
    mm_edge = mm_edge_pct / 100
    n = int(mm_n)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ---- Input grid ---------------------------------------------------------
    if mm_century:
        head_cols = st.columns([0.4, 3, 3, 1.2, 1.2])
        labels = ["#", "PLAYER 1", "PLAYER 2", "FIRST TO", "O/U LINE"]
    else:
        head_cols = st.columns([0.4, 3, 3, 1.2])
        labels = ["#", "PLAYER 1", "PLAYER 2", "FIRST TO"]
    for c, lab in zip(head_cols, labels):
        c.markdown(f"<div style='font-size:9px;letter-spacing:2px;color:#444455;"
                   f"font-family:monospace;padding-top:8px;'>{lab}</div>",
                   unsafe_allow_html=True)

    mm_inputs = []
    for i in range(n):
        if mm_century:
            cc = st.columns([0.4, 3, 3, 1.2, 1.2])
        else:
            cc = st.columns([0.4, 3, 3, 1.2])
        cc[0].markdown(
            f"<div style='font-family:monospace;color:#4fc3f7;font-weight:700;"
            f"padding-top:12px;'>{i+1}</div>", unsafe_allow_html=True)
        pa_i = cc[1].selectbox("p1", sorted_players, index=0,
                               key=f"mm_p1_{i}", label_visibility="collapsed")
        pb_i = cc[2].selectbox("p2", sorted_players, index=min(1, len(sorted_players)-1),
                               key=f"mm_p2_{i}", label_visibility="collapsed")
        ft_i = cc[3].number_input("ft", min_value=1, max_value=18, value=6, step=1,
                                  key=f"mm_ft_{i}", label_visibility="collapsed")
        line_i = 1.5
        if mm_century:
            line_i = cc[4].number_input("line", min_value=0.5, value=1.5, step=1.0,
                                        key=f"mm_line_{i}", label_visibility="collapsed",
                                        format="%.1f")
        mm_inputs.append((pa_i, pb_i, int(ft_i), float(line_i)))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN PREDICTIONS", key="mm_run"):
        rows = []
        for i, (pa_i, pb_i, ft_i, line_i) in enumerate(mm_inputs):
            row = {"idx": i + 1, "pa": pa_i, "pb": pb_i, "ft": ft_i, "line": line_i}
            if pa_i == pb_i:
                row["error"] = "same player"
            else:
                res = predict_match(pa_i, pb_i, ft_i, mm_elob_w, mm_edge,
                                    ratings_elob, ratings_elof)
                row["tp1"] = res["edge_a"]
                row["tp2"] = res["edge_b"]
                if mm_century:
                    cen = predict_match_centuries(pa_i, pb_i, ft_i, player_rates,
                                                  res["pfa_blended"])
                    ou = over_under(cen, "match", line_i)
                    row["over_tp"] = (1 / ou["over"]) * (1 + mm_edge) if ou["over"] > 0 else None
                    row["under_tp"] = (1 / ou["under"]) * (1 + mm_edge) if ou["under"] > 0 else None
            rows.append(row)
        st.session_state["mm_results"] = rows
        st.session_state["mm_results_century"] = mm_century

    # ---- Output table -------------------------------------------------------
    results = st.session_state.get("mm_results")
    if results:
        century_out = st.session_state.get("mm_results_century", False)

        def _tp(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—"

        if century_out:
            header = (
                "<table class='bet-table'><thead><tr>"
                "<th>#</th><th>PLAYER 1</th><th>PLAYER 2</th><th>FT</th>"
                "<th>TGT P1</th><th>TGT P2</th><th>LINE</th><th>OVER</th><th>UNDER</th>"
                "</tr></thead><tbody>"
            )
        else:
            header = (
                "<table class='bet-table'><thead><tr>"
                "<th>#</th><th>PLAYER 1</th><th>PLAYER 2</th><th>FT</th>"
                "<th>TGT P1</th><th>TGT P2</th>"
                "</tr></thead><tbody>"
            )

        body = ""
        for r in results:
            if r.get("error"):
                span = 5 if not century_out else 8
                body += (
                    f"<tr><td>{r['idx']}</td>"
                    f"<td style='color:#4fc3f7;'>{r['pa']}</td>"
                    f"<td style='color:#ef9a9a;'>{r['pb']}</td>"
                    f"<td colspan='{span}' style='color:#cd7f32;'>⚠ pick two different players</td></tr>"
                )
                continue
            common = (
                f"<tr><td>{r['idx']}</td>"
                f"<td style='color:#4fc3f7;font-weight:700;'>{r['pa']}</td>"
                f"<td style='color:#ef9a9a;font-weight:700;'>{r['pb']}</td>"
                f"<td>{r['ft']}</td>"
                f"<td style='color:#4fc3f7;font-weight:700;'>{_tp(r['tp1'])}</td>"
                f"<td style='color:#ef9a9a;font-weight:700;'>{_tp(r['tp2'])}</td>"
            )
            if century_out:
                ln = r["line"]
                common += (
                    f"<td style='color:#888888;'>{ln:.1f}</td>"
                    f"<td style='color:#b39ddb;font-weight:700;'>O{ln:.1f}&nbsp;&nbsp;{_tp(r.get('over_tp'))}</td>"
                    f"<td style='color:#b39ddb;font-weight:700;'>U{ln:.1f}&nbsp;&nbsp;{_tp(r.get('under_tp'))}</td>"
                )
            common += "</tr>"
            body += common

        st.markdown(header + body + "</tbody></table>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;color:#444455;font-family:monospace;margin-top:8px;'>"
            "Target price = true odds × (1 + edge). Centuries O/U is on the match total.</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:center;color:#444455;font-family:monospace;"
            "font-size:13px;padding:24px 0;'>Enter your matches and press RUN PREDICTIONS</div>",
            unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB — Outright  (Monte Carlo bracket simulation, live rating updates)
# ────────────────────────────────────────────────────────────────────

def _outright_table_html(prob_dict: dict, edge_val: float) -> str:
    rows = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    rows_html = ""
    for player, prob in rows:
        if prob <= 0:
            true_o_s, tgt_s = "—", "—"
        else:
            true_o = 1 / prob
            tgt = true_o * (1 + edge_val)
            true_o_s, tgt_s = f"{true_o:.2f}", f"{tgt:.2f}"
        rows_html += (
            f"<tr><td style='color:#ffffff;font-weight:600;font-family:Rajdhani,sans-serif;font-size:15px;'>{player}</td>"
            f"<td style='text-align:right;'>{prob*100:.2f}%</td>"
            f"<td style='text-align:right;color:#888888;'>{true_o_s}</td>"
            f"<td style='text-align:right;color:#4fc3f7;font-weight:700;'>{tgt_s}</td>"
            f"</tr>"
        )
    return (
        "<table class='rank-table'><thead><tr>"
        "<th>PLAYER</th><th class='num'>PROB</th><th class='num'>TRUE</th><th class='num'>TARGET</th>"
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )

def _outright_slot_keys(x, slot):
    return (f"out_qtog_{x}_{slot}", f"out_p_{x}_{slot}", f"out_qa_{x}_{slot}", f"out_qb_{x}_{slot}")


def _load_outright_bracket_into_state(definition: dict):
    """Programmatically sets every widget's session_state entry from a saved bracket, then
    reruns so the widgets pick the loaded values up as their initial state. Any player name
    no longer present in sorted_players (e.g. renamed in a later ratings update) is skipped
    with a warning rather than crashing the load."""
    x = definition["x_players"]
    unknown = set()

    def _safe(name):
        if name in sorted_players:
            return name
        unknown.add(name)
        return sorted_players[0]

    st.session_state["out_x"] = x
    for r, ft in enumerate(definition.get("first_to_by_round", [])):
        st.session_state[f"out_ft_{x}_{r}"] = int(ft)
    qft = definition.get("qualifier_first_to")
    if qft is not None:
        st.session_state[f"out_qft_{x}"] = int(qft)

    for slot, s in enumerate(definition.get("slots", [])):
        tog_key, p_key, qa_key, qb_key = _outright_slot_keys(x, slot)
        if s.get("mode") == "qualifier":
            st.session_state[tog_key] = True
            st.session_state[qa_key] = _safe(s["player_a"])
            st.session_state[qb_key] = _safe(s["player_b"])
        else:
            st.session_state[tog_key] = False
            st.session_state[p_key] = _safe(s["player"])

    st.session_state["outright_result"] = None
    if unknown:
        st.session_state["out_load_warning"] = (
            "Loaded, but these players are no longer in the ratings list and were reset: "
            + ", ".join(sorted(unknown))
        )
    st.rerun()


with tab_outright:
    st.markdown("### 🔱 Outright")
    st.markdown(
        "<div style='font-size:11px;color:#888888;font-family:monospace;margin-bottom:12px;'>"
        "Monte Carlo knockout-bracket simulation. Ratings update match-by-match within each "
        "simulated run of the tournament. Uses the ELOb weight and edge target set at the top of the page."
        "</div>", unsafe_allow_html=True)

    # ---- Load a saved bracket -------------------------------------------
    st.markdown("<div class='section-cap'>SAVED BRACKETS</div>", unsafe_allow_html=True)
    saved_brackets = load_saved_outright_brackets()
    lc1, lc2 = st.columns([3, 1])
    with lc1:
        load_choice = st.selectbox("Load a saved bracket", ["— none —"] + sorted(saved_brackets.keys()),
                                   key="out_load_choice", label_visibility="collapsed")
    with lc2:
        if st.button("LOAD", key="out_load_btn", disabled=(load_choice == "— none —"), use_container_width=True):
            _load_outright_bracket_into_state(saved_brackets[load_choice])
    if st.session_state.get("out_load_warning"):
        st.warning(st.session_state.pop("out_load_warning"))

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    x_players = st.selectbox("NUMBER OF PLAYERS", [2, 4, 8, 16, 32, 64, 128], index=3, key="out_x")
    out_rounds = outright_round_plan(x_players)
    n_out_rounds = len(out_rounds)

    st.markdown("<div class='section-cap'>ROUND FORMAT (FIRST TO)</div>", unsafe_allow_html=True)
    ft_cols = st.columns(n_out_rounds)
    out_first_to = []
    for i, (col, rname) in enumerate(zip(ft_cols, out_rounds)):
        with col:
            ft_i = st.selectbox(rname, list(range(1, 19)), index=9, key=f"out_ft_{x_players}_{i}")
            out_first_to.append(int(ft_i))

    st.markdown("<div class='section-cap' style='margin-top:16px;'>ROUND 1 MATCHUPS</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:8px;'>"
        "Tick \"qualifier pending\" on a side that hasn't been decided yet — enter the two players "
        "still to play it, and their winner carries their (post-qualifier) rating into Round 1.</div>",
        unsafe_allow_html=True)

    n_out_matches = x_players // 2
    out_bracket = [None] * x_players          # primary occupant per slot
    out_qual_pending = []                      # list of (slot, other_player_name)

    def _outright_side_input(slot, label, default_idx):
        tog_key, p_key, qa_key, qb_key = _outright_slot_keys(x_players, slot)
        pending = st.checkbox("qualifier pending", key=tog_key)
        if pending:
            qa = st.selectbox(f"{label} — Qualifier A", sorted_players, index=default_idx, key=qa_key)
            qb = st.selectbox(f"{label} — Qualifier B", sorted_players,
                              index=min(default_idx + 1, len(sorted_players) - 1), key=qb_key)
            out_bracket[slot] = qa
            out_qual_pending.append((slot, qb))
        else:
            p = st.selectbox(label, sorted_players, index=default_idx, key=p_key)
            out_bracket[slot] = p

    for i in range(n_out_matches):
        c1, cvs, c2 = st.columns([4, 1, 4])
        d1 = min(2 * i, len(sorted_players) - 1)
        d2 = min(2 * i + 1, len(sorted_players) - 1)
        with c1:
            _outright_side_input(2 * i, f"Match {i+1} — Player 1", d1)
        with cvs:
            st.markdown("<div style='text-align:center;padding-top:30px;color:#444455;font-family:monospace;'>vs</div>",
                       unsafe_allow_html=True)
        with c2:
            _outright_side_input(2 * i + 1, f"Match {i+1} — Player 2", d2)

    out_qualifier_ft = None
    if out_qual_pending:
        out_qualifier_ft = st.selectbox(
            "QUALIFIER FORMAT (FIRST TO) — applies to every pending qualifier above",
            list(range(1, 19)), index=5, key=f"out_qft_{x_players}")

    out_all_players = list(out_bracket) + [p for _, p in out_qual_pending]
    out_dupes = sorted({p for p in out_all_players if out_all_players.count(p) > 1})
    if out_dupes:
        st.error(f"Duplicate player(s) in the bracket: {', '.join(out_dupes)}. Each player can only occupy one slot.")

    # ---- Save this bracket ------------------------------------------------
    st.markdown("<div class='section-cap' style='margin-top:16px;'>SAVE THIS BRACKET</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        save_name = st.text_input("Bracket name", key="out_save_name",
                                  placeholder="e.g. World Championship 2027", label_visibility="collapsed")
    with sc2:
        save_clicked = st.button("SAVE", key="out_save_btn",
                                 disabled=(not save_name.strip() or bool(out_dupes)), use_container_width=True)
    if save_clicked:
        qual_slots = {slot: other for slot, other in out_qual_pending}
        definition = {
            "x_players": x_players,
            "first_to_by_round": out_first_to,
            "qualifier_first_to": out_qualifier_ft,
            "slots": [
                {"mode": "qualifier", "player_a": out_bracket[s], "player_b": qual_slots[s]}
                if s in qual_slots else {"mode": "confirmed", "player": out_bracket[s]}
                for s in range(x_players)
            ],
        }
        try:
            save_outright_bracket(save_name.strip(), definition)
            st.success(f"Saved bracket '{save_name.strip()}'.")
        except Exception as e:
            st.error(f"Could not save bracket: {e}")

    st.markdown("<div class='section-cap' style='margin-top:16px;'>SIMULATIONS</div>", unsafe_allow_html=True)
    n_sims_raw = st.number_input("Number of simulations (1 – 100,000)", min_value=1, max_value=100000,
                                 value=20000, step=1000, key="out_nsims")
    out_n_sims = int(max(1, min(100000, n_sims_raw)))

    st.markdown("<br>", unsafe_allow_html=True)
    run_outright = st.button("RUN SIMULATION", key="out_run", disabled=bool(out_dupes))

    if run_outright:
        out_roster = list(out_bracket)
        out_qualifiers = []
        for slot, other in out_qual_pending:
            out_qualifiers.append({"slot": slot, "other_idx": len(out_roster), "first_to": out_qualifier_ft})
            out_roster.append(other)

        out_progress = st.progress(0.0, text="Simulating…")
        def _out_cb(done, total):
            out_progress.progress(done / total, text=f"Simulated {done:,} / {total:,} tournaments…")
        out_result = run_outright_simulation(
            out_roster, x_players, out_qualifiers, out_first_to, out_n_sims, elob_w,
            ratings_elob, ratings_elof, progress_cb=_out_cb)
        out_progress.empty()
        st.session_state["outright_result"] = out_result

    out_result = st.session_state.get("outright_result")
    if out_result is None:
        st.markdown('<div class="match-info">Set up the bracket and press RUN SIMULATION</div>', unsafe_allow_html=True)
    else:
        sub_out_win, sub_out_final = st.tabs(["🏆  OUTRIGHT WIN", "🥈  MAKE THE FINAL"])
        with sub_out_win:
            st.markdown(
                f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:8px;'>"
                f"{out_result['n_sims']:,} simulated tournaments &nbsp;·&nbsp; edge target {edge:.1%}</div>",
                unsafe_allow_html=True)
            st.markdown(_outright_table_html(out_result["champion_prob"], edge), unsafe_allow_html=True)
        with sub_out_final:
            st.markdown(
                f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:8px;'>"
                f"{out_result['n_sims']:,} simulated tournaments &nbsp;·&nbsp; edge target {edge:.1%}</div>",
                unsafe_allow_html=True)
            st.markdown(_outright_table_html(out_result["finalist_prob"], edge), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB 6 — Rankings
# ────────────────────────────────────────────────────────────────────

def _rankings_val(value: float, color: str) -> str:
    return f"<span style='color:{color};font-weight:700;font-size:14px;'>{value:.1f}</span>"

def _cen_val(value: float, color: str) -> str:
    return f"<span style='color:{color};font-weight:700;font-size:14px;'>{value*100:.1f}%</span>"

with tab_rankings:
    rc1, rc2 = st.columns(2)
    with rc1:
        top_n = st.slider("Top N players", min_value=5, max_value=300, value=32, step=1, key="rank_n")
    with rc2:
        min_matches = st.slider("Min matches played", min_value=0, max_value=500, value=100, step=10, key="rank_min")

    st.markdown("<br>", unsafe_allow_html=True)

    rank_rows = []
    for player, info in ratings_elob.items():
        mp = info.get("matches_played", 0)
        if mp < min_matches:
            continue
        rating = info.get("rating", 0)
        cen_wf = player_rates.get(player, None)
        if cen_wf is not None and (math.isnan(cen_wf) if isinstance(cen_wf, float) else False):
            cen_wf = None
        rank_rows.append({
            "player": player,
            "matches": mp,
            "elob": rating,
            "cen_wf": cen_wf if cen_wf is not None else 0.0,
            "cen_wf_valid": cen_wf is not None,
        })

    rank_df = pd.DataFrame(rank_rows).sort_values("elob", ascending=False).reset_index(drop=True)
    rank_df = rank_df.head(top_n)

    if rank_df.empty:
        st.markdown(
            "<div class='match-info'>No players meet the minimum matches filter.</div>",
            unsafe_allow_html=True)
    else:
        max_elob = rank_df["elob"].max()

        st.markdown(
            f"<div style='font-size:10px;color:#444455;font-family:monospace;margin-bottom:12px;'>"
            f"Showing top {len(rank_df)} players &nbsp;·&nbsp; min {min_matches} matches played"
            f"</div>",
            unsafe_allow_html=True)

        head = (
            "<table class='rank-table'><thead><tr>"
            "<th style='width:36px;'>#</th>"
            "<th>PLAYER</th>"
            "<th class='num' style='width:90px;'>MATCHES</th>"
            "<th style='width:260px;'>ELO BETA</th>"
            "<th style='width:200px;'>CEN RAT</th>"
            "</tr></thead><tbody>"
        )

        rows_html = ""
        for i, row in rank_df.iterrows():
            rank_num = i + 1
            if rank_num == 1:
                rank_color = "#ffd700"
            elif rank_num == 2:
                rank_color = "#c0c0c0"
            elif rank_num == 3:
                rank_color = "#cd7f32"
            else:
                rank_color = "#444455"
            row_style = ""
            if rank_num == 16:
                row_style = "border-bottom: 2px solid #2a2a3a;"

            matches_style = "color:#ef5350;font-weight:700;" if row["matches"] < 50 else "color:#888888;"

            elob_bar = _rankings_val(row["elob"], "#4fc3f7")
            cen_bar  = _cen_val(row["cen_wf"], "#b39ddb") if row["cen_wf_valid"] and row["cen_wf"] > 0 else (
                "<span style='color:#444455;font-size:11px;'>—</span>"
            )

            rows_html += (
                f"<tr style='{row_style}'>"
                f"<td style='color:{rank_color};font-weight:700;font-size:13px;text-align:right;padding-right:12px;'>{rank_num}</td>"
                f"<td style='color:#ffffff;font-weight:600;font-family:Rajdhani,sans-serif;font-size:15px;'>{row['player']}</td>"
                f"<td style='text-align:right;{matches_style}'>{row['matches']}</td>"
                f"<td>{elob_bar}</td>"
                f"<td>{cen_bar}</td>"
                f"</tr>"
            )

        st.markdown(head + rows_html + "</tbody></table>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# TAB 7 — Update Results  (scrape → save → run models → save)
# ────────────────────────────────────────────────────────────────────

with tab_update:
    st.markdown("### ⚙️ Update Results")
    st.markdown(
        "<div style='font-size:11px;color:#888888;font-family:monospace;margin-bottom:12px;'>"
        "Scrape new snooker results, commit them to GitHub, then re-run the models. "
        "Each save commits to the repo, which automatically reboots the app with fresh data."
        "</div>", unsafe_allow_html=True)

    token, repo_name, branch = _gh_config()

    # ---- Prerequisite checks -------------------------------------------------
    issues = []
    if not _SCRAPER_AVAILABLE:
        issues.append(f"`scraper_functions.py` / `requests` / `bs4` not importable — {_SCRAPER_ERR}")
    if not _GITHUB_AVAILABLE:
        issues.append("`PyGithub` not installed — add `PyGithub` to requirements.txt")
    if token is None:
        issues.append("GitHub secrets missing — add a `[github]` block in Streamlit Secrets (token / repo / branch)")
    import os
    if not os.path.exists(PARQUET_PATH):
        issues.append(f"`{PARQUET_PATH}` (master dataset) not found in the repo — commit it once to enable updates")

    if issues:
        st.markdown(
            "<div style='font-size:10px;letter-spacing:1px;color:#ef9a9a;font-family:monospace;margin-bottom:6px;'>"
            "SETUP NEEDED</div>", unsafe_allow_html=True)
        for it in issues:
            st.markdown(f"- {it}")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ---- Password gate -------------------------------------------------------
    try:
        cfg_pw = st.secrets["app"]["update_password"]
    except Exception:
        cfg_pw = None

    authed = True
    if cfg_pw is not None:
        if st.session_state.get("upd_authed"):
            authed = True
        else:
            authed = False
            pw = st.text_input("Update password", type="password", key="upd_pw")
            if st.button("Unlock", key="upd_unlock"):
                if pw == cfg_pw:
                    st.session_state["upd_authed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    else:
        st.markdown(
            "<div style='font-size:10px;color:#cd7f32;font-family:monospace;margin-bottom:8px;'>"
            "⚠ No update password set. Add `[app] update_password` in Secrets to protect these buttons "
            "if your app is public.</div>", unsafe_allow_html=True)

    can_commit = _GITHUB_AVAILABLE and token is not None

    if authed:
        # ===== STEP 1 — Scrape & compare =====================================
        st.markdown("<div class='upd-step'>① SCRAPE &amp; COMPARE</div>", unsafe_allow_html=True)
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            seasons_raw = st.text_input(
                "Seasons to scrape (comma-separated)", value="2026, 2027", key="upd_seasons",
                help="Season N means the YYYY ending year, e.g. 2026 = 2025-2026 season.")
        with sc2:
            default_min_date = (pd.Timestamp.now() - pd.DateOffset(months=1)).date()
            min_date_val = st.date_input(
                "Detect matches on/after", value=default_min_date, key="upd_min_date",
                help="Only games on or after this date are flagged as new. Defaults to one month ago.")

        if st.button("RUN SCRAPE", key="btn_scrape", disabled=not _SCRAPER_AVAILABLE):
            try:
                seasons = [int(s.strip()) for s in seasons_raw.split(",") if s.strip()]
            except ValueError:
                seasons = []
                st.error("Could not parse seasons. Use e.g. `2026, 2027`.")

            min_date_str = min_date_val.strftime("%Y-%m-%d")
            if seasons:
                with st.status("Scraping cuetracker…", expanded=True) as status:
                    try:
                        combined, new_matches, preview = scrape_and_diff(
                            seasons, min_date=min_date_str,
                            progress_cb=lambda m: status.write(m))
                        # Stage locally so Save / Run-models use fresh data this session
                        combined.to_parquet(PARQUET_PATH)
                        st.session_state["upd_preview"] = preview
                        st.session_state["upd_new_count"] = len(new_matches)
                        st.session_state["upd_scraped"] = True
                        status.update(
                            label=f"Done — {len(new_matches)} new match(es) staged locally",
                            state="complete")
                    except Exception as e:
                        st.session_state["upd_scraped"] = False
                        status.update(label="Scrape failed", state="error")
                        st.exception(e)

        if st.session_state.get("upd_preview") is not None:
            n = st.session_state.get("upd_new_count", 0)
            st.markdown(
                f"<div style='font-size:12px;color:#a5d6a7;font-family:monospace;margin:8px 0;'>"
                f"{n} new match(es) found vs the saved dataset:</div>", unsafe_allow_html=True)
            if n > 0:
                st.dataframe(st.session_state["upd_preview"], use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    "<div style='font-size:11px;color:#888888;font-family:monospace;'>"
                    "Nothing new — the dataset is already up to date.</div>", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ===== STEP 2 — Save new data to GitHub ==============================
        st.markdown("<div class='upd-step'>② SAVE NEW MATCH DATA → GITHUB</div>", unsafe_allow_html=True)
        save_data_disabled = not (can_commit and st.session_state.get("upd_scraped"))
        if st.button("SAVE DATA TO GITHUB", key="btn_save_data", disabled=save_data_disabled):
            with st.status("Committing match data…", expanded=True) as status:
                try:
                    status.write(f"Reading staged {PARQUET_PATH}…")
                    with open(PARQUET_PATH, "rb") as f:
                        pq_bytes = f.read()
                    n = st.session_state.get("upd_new_count", 0)
                    msg = f"Update match data ({n} new matches) via app {datetime.utcnow():%Y-%m-%d %H:%M UTC}"
                    status.write("Pushing commit to GitHub…")
                    sha = commit_files_to_github(token, repo_name, branch, {PARQUET_PATH: pq_bytes}, msg)
                    status.update(label=f"Committed ({sha[:7]}). App will reboot shortly.", state="complete")
                    st.success(f"Pushed {PARQUET_PATH} → {repo_name}@{branch}  ({sha[:7]})")
                    st.info("Streamlit will redeploy automatically on the new commit. "
                            "You can run the models now (same session) or after it reboots.")
                except Exception as e:
                    status.update(label="Commit failed", state="error")
                    st.exception(e)
        if save_data_disabled and not st.session_state.get("upd_scraped"):
            st.markdown("<div style='font-size:10px;color:#444455;font-family:monospace;'>"
                        "Run a scrape first.</div>", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ===== STEP 3 — Run models & save ====================================
        st.markdown("<div class='upd-step'>③ RUN MODELS &amp; SAVE → GITHUB</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;color:#888888;font-family:monospace;margin-bottom:6px;'>"
            "Runs ELO-beta, ELO-frames and the centuries model on the current dataset, then commits "
            "ratings.json, player_rates.json and player_matches_df.csv. This is the heavy step "
            "(~minutes, loads the full frame dataset).</div>", unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            elob_k = st.number_input("ELO-beta K", value=14, step=1, key="upd_elob_k")
        with mc2:
            elof_k = st.number_input("ELO-frames K", value=16, step=1, key="upd_elof_k")
        with mc3:
            halflife = st.number_input("Century halflife (days)", value=220, step=10, key="upd_halflife")

        run_models_disabled = not (can_commit and os.path.exists(PARQUET_PATH))
        if st.button("RUN MODELS & SAVE", key="btn_run_models", disabled=run_models_disabled):
            with st.status("Running models…", expanded=True) as status:
                try:
                    status.write(f"Loading {PARQUET_PATH}…")
                    df_frames = pd.read_parquet(PARQUET_PATH)

                    ratings_obj, new_player_rates, matches_df = run_models_pipeline(
                        df_frames, elob_k=int(elob_k), elof_k=int(elof_k),
                        halflife=int(halflife), progress_cb=lambda m: status.write(m))

                    # Serialize (allow_nan=True keeps NaN tokens, matching existing files)
                    ratings_bytes = json.dumps(ratings_obj).encode("utf-8")
                    rates_bytes = json.dumps(new_player_rates).encode("utf-8")
                    csv_bytes = matches_df.to_csv(index=False).encode("utf-8")

                    # Write locally too so this session reflects new data after cache clear
                    status.write("Writing local copies…")
                    with open(RATINGS_PATH, "wb") as f:
                        f.write(ratings_bytes)
                    with open(PLAYER_RATES_PATH, "wb") as f:
                        f.write(rates_bytes)
                    with open(MATCHES_CSV_PATH, "wb") as f:
                        f.write(csv_bytes)

                    status.write("Committing model files to GitHub…")
                    msg = f"Re-run models via app {datetime.utcnow():%Y-%m-%d %H:%M UTC}"
                    sha = commit_files_to_github(
                        token, repo_name, branch,
                        {RATINGS_PATH: ratings_bytes,
                         PLAYER_RATES_PATH: rates_bytes,
                         MATCHES_CSV_PATH: csv_bytes},
                        msg)

                    st.cache_data.clear()
                    status.update(
                        label=f"Models committed ({sha[:7]}). App will reboot shortly.",
                        state="complete")
                    st.success(
                        f"Pushed ratings.json, player_rates.json, player_matches_df.csv → "
                        f"{repo_name}@{branch}  ({sha[:7]})")
                    st.info("Caches cleared. The app will redeploy on the new commit; "
                            "predictions will then use the updated ratings.")
                except MemoryError:
                    status.update(label="Out of memory", state="error")
                    st.error(
                        "The model step ran out of memory on this host. The scrape/save steps still "
                        "work — run the models locally (match_models_update + century_functions) and "
                        "commit those three files, or run the app on a larger instance.")
                except Exception as e:
                    status.update(label="Model run failed", state="error")
                    st.exception(e)
