import json
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
    div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Model functions ───────────────────────────────────────────────────
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
    )


# ── Load ratings ──────────────────────────────────────────────────────
@st.cache_data
def load_ratings():
    with open("ratings.json", encoding="utf-8") as f:
        data = json.load(f)
    return data["elob"], data["elof"]

ratings_elob, ratings_elof = load_ratings()

sorted_players = sorted(
    ratings_elob.keys(),
    key=lambda p: ratings_elob[p]["rating"],
    reverse=True
)


# ── Header ────────────────────────────────────────────────────────────
st.markdown("## 🎱 Match Predictor")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────
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
                       format="%.0f%%",
                       help="Weight given to ELO-beta vs ELO-frames")
with sl3:
    edge = st.slider("EDGE TARGET", 0.0, 0.20, 0.05, step=0.005,
                     format="%.1f%%")

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("RUN PREDICTION")


# ── Results ───────────────────────────────────────────────────────────
if run:
    if player_a == player_b:
        st.error("Please select two different players.")
        st.stop()

    r = predict_match(player_a, player_b, first_to,
                      elob_w, edge, ratings_elob, ratings_elof)

    # Match info strip
    st.markdown(
        f'<div class="match-info">'
        f'First to {first_to} &nbsp;·&nbsp; Best of {r["best_of"]} &nbsp;·&nbsp; '
        f'ELOb {elob_w:.0%} / ELOf {1-elob_w:.0%}'
        f'</div>',
        unsafe_allow_html=True
    )

    card_a, card_b = st.columns(2)

    def render_card(col, name, prob, true_o, edge_p,
                    elob_r, elof_r,
                    elob_pf, elob_pm,
                    elof_pf, elof_pm,
                    m_elob, m_elof,
                    color, bar_class, card_class, edge_pct):
        with col:
            bar_w = int(prob * 100)
            st.markdown(f"""
            <div class="{card_class}">
                <div class="player-name-{'a' if color == '#4fc3f7' else 'b'}">{name}</div>
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
                    <div class="metric-cap">TARGET (+{edge_pct:.1%} EDGE)</div>
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

    render_card(
        card_a, player_a,
        r["prob_a"], r["true_a"], r["edge_a"],
        r["elob_ra"], r["elof_ra"],
        r["elob_pfa"], r["elob_pa"],
        r["elof_pfa"], r["elof_pa"],
        r["elob_ma"], r["elof_ma"],
        "#4fc3f7", "prob-bar-fill-a", "card-a", edge
    )
    render_card(
        card_b, player_b,
        r["prob_b"], r["true_b"], r["edge_b"],
        r["elob_rb"], r["elof_rb"],
        r["elob_pfb"], r["elob_pb"],
        r["elof_pfb"], r["elof_pb"],
        r["elob_mb"], r["elof_mb"],
        "#ef9a9a", "prob-bar-fill-b", "card-b", edge
    )
