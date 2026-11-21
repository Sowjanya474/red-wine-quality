import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Red Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")


bundle = load_bundle()
model = bundle["model"]
threshold = float(bundle["threshold"])
features = bundle["features"]

# name: (min, max, typical value, step, display format, plain-language hint)
RANGES = {
    "fixed acidity": (4.6, 15.9, 7.9, 0.1, "%.1f", "Mostly tartaric acid, in g/dm³. Gives the wine its tartness."),
    "volatile acidity": (0.12, 1.58, 0.52, 0.01, "%.2f", "Mostly acetic acid, in g/dm³. High levels smell like vinegar."),
    "citric acid": (0.0, 1.0, 0.26, 0.01, "%.2f", "In g/dm³. Adds freshness."),
    "pH": (2.74, 4.01, 3.31, 0.01, "%.2f", "Lower means more acidic."),
    "alcohol": (8.4, 14.9, 10.2, 0.1, "%.1f", "Percent by volume."),
    "density": (0.99007, 1.00369, 0.9968, 0.0001, "%.4f", "In g/cm³. Drops as alcohol rises."),
    "residual sugar": (0.9, 15.5, 2.2, 0.1, "%.1f", "Sugar left after fermentation, in g/dm³."),
    "sulphates": (0.33, 2.0, 0.62, 0.01, "%.2f", "Potassium sulphate, in g/dm³. Acts as a preservative."),
    "chlorides": (0.012, 0.611, 0.079, 0.001, "%.3f", "Salt content, in g/dm³."),
    "free sulfur dioxide": (1.0, 72.0, 14.0, 1.0, "%.0f", "In mg/dm³. Protects the wine from spoiling."),
    "total sulfur dioxide": (6.0, 289.0, 38.0, 1.0, "%.0f", "In mg/dm³. Free plus bound sulfur dioxide."),
}

GROUPS = [
    ("Acidity", ["fixed acidity", "volatile acidity", "citric acid", "pH"]),
    ("Body", ["alcohol", "density", "residual sugar"]),
    ("Preservatives and salts", ["sulphates", "chlorides", "free sulfur dioxide", "total sulfur dioxide"]),
]

PRESET_GOOD = {"alcohol": 13.0, "sulphates": 0.90, "volatile acidity": 0.40, "citric acid": 0.40}
PRESET_POOR = {"alcohol": 9.0, "sulphates": 0.50, "volatile acidity": 0.90, "citric acid": 0.05}


def label(name):
    return name if name == "pH" else name.capitalize()


def html(block):
    """Collapse an HTML snippet onto one line so Markdown never treats it as code."""
    return " ".join(line.strip() for line in block.splitlines() if line.strip())


def apply_preset(overrides):
    for n, spec in RANGES.items():
        st.session_state[f"f_{n}"] = float(overrides.get(n, spec[2]))


for n, spec in RANGES.items():
    st.session_state.setdefault(f"f_{n}", float(spec[2]))

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Manrope:wght@400;500;700&display=swap');

:root {
  --claret: #5B1A2E;
  --cellar: #1C0B14;
  --mist: #F5EEF1;
  --ink: #2B2330;
  --teal: #7FD1BF;
  --rose: #E58AA3;
}
html, body, .stApp, .stApp p, .stApp label, .stApp button {
  font-family: 'Manrope', sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
footer, #MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.app-title {
  font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.7rem;
  line-height: 1.1; letter-spacing: -0.01em; color: var(--claret); margin: 0 0 .5rem;
}
.app-sub { max-width: 62ch; color: var(--ink); opacity: .82; font-size: 1.04rem; margin: 0 0 1.4rem; }

.group-title {
  font-family: 'Fraunces', serif; font-size: 1.3rem; color: var(--claret);
  margin: 1.4rem 0 .5rem; padding-bottom: .35rem; border-bottom: 1px solid #D9C4CC;
}
[data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--ink); }

.stButton > button {
  border-radius: 999px; border: 1.5px solid var(--claret); color: var(--claret);
  background: transparent; font-weight: 600; padding: .3rem 1rem;
}
.stButton > button:hover { background: var(--claret); color: #fff; border-color: var(--claret); }
.stButton > button:focus-visible { outline: 3px solid #B88A9A; outline-offset: 2px; }

.card { background: var(--cellar); color: var(--mist); border-radius: 20px; padding: 1.6rem 1.6rem 1.4rem; }
.top { display: flex; gap: 1.3rem; align-items: center; flex-wrap: wrap; }
.pct { font-family: 'Fraunces', serif; font-size: 3.7rem; line-height: 1; font-weight: 600; }
.pct-note { font-size: .88rem; opacity: .78; margin: .3rem 0 .8rem; max-width: 22ch; }
.pill { display: inline-block; padding: .32rem .85rem; border-radius: 999px; font-weight: 700; font-size: .92rem; }
.pill.good { background: var(--teal); color: #0E2A25; }
.pill.not { background: #3A2630; color: #EBDCE2; border: 1px solid #5A3E4B; }
.rule { border-top: 1px solid #3A2630; margin: 1.3rem 0 1rem; }
.drv-head { font-family: 'Fraunces', serif; font-size: 1.1rem; margin-bottom: .15rem; }
.drv-sub { font-size: .82rem; opacity: .72; margin-bottom: .8rem; }
.drv { margin-bottom: .75rem; }
.drv-row { display: flex; justify-content: space-between; font-size: .92rem; }
.drv-row .up { color: var(--teal); font-weight: 700; }
.drv-row .down { color: var(--rose); font-weight: 700; }
.drv-track { height: 6px; background: #2E1B25; border-radius: 3px; margin-top: .3rem; }
.drv-bar { height: 6px; border-radius: 3px; }
.drv-bar.up { background: var(--teal); }
.drv-bar.down { background: var(--rose); }
.fine { font-size: .8rem; opacity: .68; margin-top: 1.1rem; }
.foot { color: #6B5A63; font-size: .86rem; margin-top: 2rem; max-width: 80ch; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- header
st.markdown(
    html(
        """
<h1 class="app-title">Red Wine Quality Predictor</h1>
<p class="app-sub">Describe a red wine with the sliders. The glass fills with the chance that
tasters would rate it 7 or higher out of 10.</p>
"""
    ),
    unsafe_allow_html=True,
)

pc = st.columns([1, 1, 1, 2])
pc[0].button("Typical wine", on_click=apply_preset, args=({},), use_container_width=True)
pc[1].button("Likely good", on_click=apply_preset, args=(PRESET_GOOD,), use_container_width=True)
pc[2].button("Likely poor", on_click=apply_preset, args=(PRESET_POOR,), use_container_width=True)

left, right = st.columns([1.25, 1], gap="large")

# ---------------------------------------------------------------- inputs
with left:
    for title, names in GROUPS:
        st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        shown = [n for n in names if n in features]
        for i, n in enumerate(shown):
            lo, hi, _, step, fmt, hint = RANGES[n]
            with cols[i % 2]:
                st.slider(
                    label(n),
                    min_value=float(lo),
                    max_value=float(hi),
                    step=float(step),
                    format=fmt,
                    help=hint,
                    key=f"f_{n}",
                )

# ---------------------------------------------------------------- result
values = {n: float(st.session_state[f"f_{n}"]) for n in features}

# One batch: the wine as entered, then the same wine with each feature set to typical.
rows = [dict(values)]
for n in features:
    r = dict(values)
    r[n] = RANGES[n][2]
    rows.append(r)
probs = model.predict_proba(pd.DataFrame(rows)[features])[:, 1]
p = float(probs[0])
effects = {n: p - float(probs[i + 1]) for i, n in enumerate(features)}

is_good = p >= threshold
pct_text = f"{p * 100:.1f}%" if p < 0.10 else f"{p * 100:.0f}%"
verdict = "Likely rated 7 or higher" if is_good else "Probably rated below 7"
pill_class = "good" if is_good else "not"

# Glass geometry: bowl spans y = 8 (rim) to y = 106 (bottom)
fill = max(p, 0.03)
y_top = 106 - fill * 98
y_thr = 106 - threshold * 98
bowl = "M22 8 H98 C98 70 84 104 60 106 C36 104 22 70 22 8 Z"
glass = html(
    f"""
<svg viewBox="0 0 120 200" width="112" role="img" aria-label="Wine glass filled to {pct_text}">
  <defs>
    <clipPath id="bowl"><path d="{bowl}"/></clipPath>
    <linearGradient id="wine" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#A3314F"/><stop offset="1" stop-color="#5B1A2E"/>
    </linearGradient>
  </defs>
  <g clip-path="url(#bowl)">
    <rect x="0" y="{y_top:.1f}" width="120" height="120" fill="url(#wine)"/>
    <line x1="0" x2="120" y1="{y_thr:.1f}" y2="{y_thr:.1f}" stroke="#7FD1BF" stroke-width="1.6" stroke-dasharray="3 3"/>
  </g>
  <path d="{bowl}" fill="none" stroke="#E9D9DF" stroke-width="2"/>
  <rect x="58" y="106" width="4" height="78" fill="#E9D9DF"/>
  <ellipse cx="60" cy="185" rx="30" ry="5" fill="#E9D9DF"/>
</svg>
"""
)

top = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
top = [(n, e) for n, e in top if abs(e) >= 0.005][:4]

if top:
    biggest = max(abs(e) for _, e in top)
    drivers = ""
    for n, e in top:
        direction = "up" if e > 0 else "down"
        width = max(6, abs(e) / biggest * 100)
        drivers += html(
            f"""
<div class="drv">
  <div class="drv-row"><span>{label(n)} {RANGES[n][4] % values[n]}</span>
  <span class="{direction}">{e * 100:+.0f} pts</span></div>
  <div class="drv-track"><div class="drv-bar {direction}" style="width:{width:.0f}%"></div></div>
</div>
"""
        )
    drivers_block = (
        '<div class="drv-head">What is moving the result</div>'
        '<div class="drv-sub">Each measure compared with a typical wine, in percentage points.</div>'
        + drivers
    )
else:
    drivers_block = (
        '<div class="drv-head">What is moving the result</div>'
        '<div class="drv-sub">This wine is close to typical on every measure.</div>'
    )

with right:
    st.markdown(
        html(
            f"""
<div class="card">
  <div class="top">
    {glass}
    <div>
      <div class="pct">{pct_text}</div>
      <div class="pct-note">chance this wine is rated 7 or higher</div>
      <span class="pill {pill_class}">{verdict}</span>
    </div>
  </div>
  <div class="rule"></div>
  {drivers_block}
  <div class="fine">The dashed line on the glass is the model’s cut-off:
  a wine counts as good at {threshold * 100:.0f}% or higher.</div>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="foot">Trained on 1,359 unique red Vinho Verde wines (UCI Wine Quality dataset). '
    "On wines it had never seen, the model found about 3 in 4 good wines (recall 0.76), "
    "but only 4 in 10 of its “good” calls were right (precision 0.41). "
    "It is a learning project and an estimate, not a tasting score.</div>",
    unsafe_allow_html=True,
)