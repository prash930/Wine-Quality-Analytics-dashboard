import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (no sidebar)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VinIQ · Wine Quality Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: #0d0f18 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide only branding */
#MainMenu, footer, header { visibility: hidden !important; }

/* Sidebar toggle arrow — always visible & styled */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    background: #a78bfa !important;
    border-radius: 0 8px 8px 0 !important;
    color: #fff !important;
    width: 1.6rem !important;
    box-shadow: 2px 0 12px rgba(167,139,250,0.4) !important;
}
[data-testid="collapsedControl"]:hover {
    background: #7c3aed !important;
}
[data-testid="collapsedControl"] svg { stroke: #fff !important; }

/* Sidebar background */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background: #12151f !important;
    border-right: 1px solid #1e2235 !important;
}
section[data-testid="stSidebar"] * { color: #c8cce0 !important; }
section[data-testid="stSidebar"] label {
    color: #8890b0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Expander – filter panel */
[data-testid="stExpander"] {
    background: #12151f !important;
    border: 1px solid #1e2235 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #c8cce0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Selectbox / multiselect */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: #1a1d2e !important;
    border: 1px solid #2a2e45 !important;
    border-radius: 8px !important;
    color: #e0e4f0 !important;
}

/* Labels */
label, .stSelectbox label, .stSlider label,
.stCheckbox label, .stRadio label {
    color: #8890b0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Sliders */
[data-testid="stSlider"] * { color: #e0e4f0 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #13162a, #1a2040) !important;
    border: 1px solid #2a2e4a !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #6b73a0 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8ecff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #6b73a0 !important;
    background: transparent !important;
    border: none !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

hr { border-color: #1e2235 !important; }

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px;
}

[data-testid="stDataFrame"] { background: #13162a !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f18; }
::-webkit-scrollbar-thumb { background: #2a2e4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4265; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA  (cached once on startup)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality.csv")
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    out = df.copy()
    for c in cols:
        q1, q3 = out[c].quantile(0.25), out[c].quantile(0.75)
        iqr = q3 - q1
        out = out[(out[c] >= q1 - 1.5*iqr) & (out[c] <= q3 + 1.5*iqr)]
    return out

df_raw   = load_data()
df_clean = clean_data(df_raw)

NUMERIC_COLS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

C = {   # colour palette
    "red":    "#e05f5f", "white":  "#a78bfa",
    "a1":     "#60a5fa", "a2":     "#f472b6",
    "bg":     "#0d0f18", "card":   "#13162a",
    "border": "#2a2e4a", "text":   "#e8ecff",
    "muted":  "#6b73a0",
}

BASE_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
    font=dict(family="Inter", color=C["text"]),
    margin=dict(l=40, r=20, t=46, b=36),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"], borderwidth=1),
    xaxis=dict(gridcolor="#1e2235", zerolinecolor="#1e2235"),
    yaxis=dict(gridcolor="#1e2235", zerolinecolor="#1e2235"),
)

def layout(**overrides):
    """Merge BASE_LAYOUT with overrides, safely."""
    base = {k: v for k, v in BASE_LAYOUT.items()
            if k not in overrides}
    return {**base, **overrides}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ─────────────────────────────────────────────
# HEADER  +  sidebar toggle button
# ─────────────────────────────────────────────
hdr_title, hdr_btn = st.columns([11, 1])
with hdr_title:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.2rem;'>"
        "<span style='font-size:2.2rem;'>🍷</span>"
        "<div><h1 style='color:#e8ecff;margin:0;font-size:1.6rem;font-weight:700;'>"
        "VinIQ &nbsp;·&nbsp; Wine Quality Analytics</h1>"
        "<p style='color:#6b73a0;margin:0;font-size:0.78rem;'>Interactive wine data explorer</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
with hdr_btn:
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    btn_label = "✕ Hide" if st.session_state.sidebar_open else "☰ Filters"
    if st.button(btn_label, key="sb_toggle", use_container_width=True):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

# ─────────────────────────────────────────────
# FILTERS  — sidebar OR inline depending on toggle
# ─────────────────────────────────────────────
def _filter_widgets():
    """Shared filter widgets. Must be called inside a container context."""
    wt  = st.selectbox("Wine Type", ["All", "Red", "White"], key="wt")
    ro  = st.checkbox("Remove Outliers (IQR)", value=False, key="ro")
    q_min, q_max = int(df_raw["quality"].min()), int(df_raw["quality"].max())
    qr  = st.slider("Quality Score Range", q_min, q_max, (q_min, q_max), key="qr")
    a_min = float(df_raw["alcohol"].min())
    a_max = float(df_raw["alcohol"].max())
    ar  = st.slider("Alcohol % Range", a_min, a_max, (a_min, a_max), step=0.1, key="ar")
    st.markdown("---")
    xf  = st.selectbox("Scatter X", NUMERIC_COLS, index=NUMERIC_COLS.index("alcohol"), key="xf")
    yf  = st.selectbox("Scatter Y", NUMERIC_COLS, index=NUMERIC_COLS.index("volatile acidity"), key="yf")
    cbq = st.checkbox("Colour by Quality", value=True, key="cbq")
    return wt, ro, qr, ar, xf, yf, cbq

if st.session_state.sidebar_open:
    # ── Actual left sidebar ──
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1rem 0 0.4rem;'>"
            "<span style='font-size:2rem;'>🍷</span>"
            "<h2 style='color:#e8ecff;margin:0.2rem 0 0;font-size:1rem;font-weight:700;'>VinIQ</h2>"
            "<p style='color:#6b73a0;font-size:0.68rem;margin:0;'>Wine Quality Analytics</p></div>"
            "<hr style='border-color:#1e2235;margin:0.5rem 0;'/>"
            "<p style='color:#8890b0;font-size:0.72rem;text-transform:uppercase;"
            "letter-spacing:0.08em;margin:0 0 0.4rem;'>📋 Filters</p>",
            unsafe_allow_html=True,
        )
        wine_type, remove_outliers, quality_range, \
        alcohol_range, x_feat, y_feat, color_by_quality = _filter_widgets()
else:
    # ── Inline expandable panel (no sidebar) ──
    with st.expander("⚙️  Filters & Settings", expanded=True):
        wine_type, remove_outliers, quality_range, \
        alcohol_range, x_feat, y_feat, color_by_quality = _filter_widgets()


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df_base = df_clean if remove_outliers else df_raw

if wine_type == "Red":
    df = df_base[df_base["color"] == "red"]
elif wine_type == "White":
    df = df_base[df_base["color"] == "white"]
else:
    df = df_base.copy()

df = df[
    (df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1]) &
    (df["alcohol"] >= alcohol_range[0]) & (df["alcohol"] <= alcohol_range[1])
]

df_red   = df[df["color"] == "red"]
df_white = df[df["color"] == "white"]

if len(df) == 0:
    st.warning("No data matches the current filters. Please adjust your selections.")
    st.stop()

# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e2235;margin:0.5rem 0 0.8rem 0;'/>", unsafe_allow_html=True)

total   = len(df)
premium = len(df[df["quality"] >= 7])
prem_pct = round(premium / total * 100, 1)
red_pct  = round(len(df_red) / total * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Samples",     f"{total:,}",             f"{total - len(df_base):+,} vs full")
k2.metric("Premium (≥ 7)",     f"{premium:,}",           f"{prem_pct}% share")
k3.metric("Avg Quality Score", f"{df['quality'].mean():.3f}", f"Max {int(df['quality'].max())}")
k4.metric("Avg Alcohol %",     f"{df['alcohol'].mean():.2f}", f"σ {df['alcohol'].std():.2f}")
k5.metric("Red / White",       f"{red_pct}% Red",        f"{100-red_pct:.1f}% White")

st.markdown("<hr style='border-color:#1e2235;margin:0.8rem 0;'/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Distributions", "🔥 Correlations", "🔬 Feature Explorer", "📋 Data"
])

# ══════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 1])

    # ── Quality bar chart ──
    with col_a:
        qc = df.groupby(["quality", "color"]).size().reset_index(name="n")
        fig = go.Figure()
        for cv, lc, off in [("red", C["red"], -0.22), ("white", C["white"], 0.22)]:
            sub = qc[qc["color"] == cv].sort_values("quality")
            if len(sub):
                fig.add_trace(go.Bar(x=sub["quality"], y=sub["n"], name=cv.capitalize(),
                                     marker_color=lc, opacity=0.87, width=0.4, offset=off))
        fig.update_layout(title="Quality Score Distribution by Wine Type",
                          **layout(barmode="overlay", xaxis_title="Quality Score", yaxis_title="Count"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Donut ──
    with col_b:
        fig2 = go.Figure(go.Pie(
            labels=["Red", "White"], values=[len(df_red), len(df_white)],
            hole=0.62,
            marker=dict(colors=[C["red"], C["white"]], line=dict(color=C["bg"], width=3)),
            textinfo="percent+label", textfont=dict(color=C["text"], size=11),
            showlegend=False,
        ))
        fig2.update_layout(
            title="Colour Split",
            **layout(xaxis=None, yaxis=None, margin=dict(l=10, r=10, t=50, b=10)),
            annotations=[dict(text=f"<b>{total:,}</b><br>wines",
                              x=0.5, y=0.5, font_size=12, font_color=C["text"], showarrow=False)],
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Violin ──
    fig3 = go.Figure()
    for cv, vc in [("red", C["red"]), ("white", C["white"])]:
        sub = df[df["color"] == cv]
        if len(sub):
            fig3.add_trace(go.Violin(
                x=sub["color"].str.capitalize(), y=sub["quality"],
                name=cv.capitalize(), line_color=vc,
                box_visible=True, meanline_visible=True,
                points="outliers", pointpos=0,
                marker=dict(color=vc, opacity=0.4, size=3),
            ))
    fig3.update_layout(title="Quality Distribution – Violin",
                       **layout(yaxis_title="Quality Score",
                                violingap=0.3, violinmode="overlay"))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Heatmap: avg feature per quality ──
    st.markdown("#### 📊 Average Feature Values by Quality Grade")
    avg_q = df.groupby("quality")[NUMERIC_COLS].mean()
    fig4 = go.Figure(go.Heatmap(
        z=avg_q.T.values, x=avg_q.index.astype(str), y=NUMERIC_COLS,
        colorscale="Viridis", colorbar=dict(tickfont=dict(color=C["text"])),
        text=np.round(avg_q.T.values, 2), texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
    ))
    fig4.update_layout(title="Mean Feature Values per Quality Grade",
                       **layout(xaxis=dict(title="Quality Score", gridcolor="#1e2235"),
                                yaxis=dict(autorange="reversed", gridcolor="#1e2235"),
                                height=380))
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – DISTRIBUTIONS
# ══════════════════════════════════════════════
with tab2:
    fc_a, fc_b = st.columns([2, 1])
    with fc_a:
        feat_hist = st.selectbox("Feature to inspect", NUMERIC_COLS, key="hist_feat")
    with fc_b:
        nbins_val = st.slider("Bins", 10, 100, 40, key="nbins")

    col2a, col2b = st.columns(2)

    with col2a:
        fig5 = go.Figure()
        for cv, vc in [("red", C["red"]), ("white", C["white"])]:
            sub = df[df["color"] == cv]
            if len(sub):
                fig5.add_trace(go.Histogram(x=sub[feat_hist], name=cv.capitalize(),
                                            marker_color=vc, opacity=0.70, nbinsx=nbins_val))
        fig5.update_layout(title=f"{feat_hist} – Distribution", barmode="overlay",
                           **layout(xaxis_title=feat_hist, yaxis_title="Count"))
        st.plotly_chart(fig5, use_container_width=True)

    with col2b:
        fig6 = go.Figure()
        cseq = px.colors.sequential.Viridis
        qs   = sorted(df["quality"].unique())
        for i, q in enumerate(qs):
            cidx = int(i / max(len(qs)-1, 1) * (len(cseq)-1))
            fig6.add_trace(go.Box(y=df[df["quality"]==q][feat_hist],
                                  name=f"Q{int(q)}", marker_color=cseq[cidx], boxmean=True))
        fig6.update_layout(title=f"{feat_hist} by Quality",
                           **layout(yaxis_title=feat_hist, showlegend=False))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("#### 📋 Descriptive Statistics")
    st.dataframe(
        df[NUMERIC_COLS + ["quality"]].describe().round(4).style.background_gradient(cmap="Blues", axis=1),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# TAB 3 – CORRELATIONS
# ══════════════════════════════════════════════
with tab3:
    corr = df[NUMERIC_COLS + ["quality"]].corr().round(3)

    fig7 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
        colorbar=dict(tickfont=dict(color=C["text"])),
    ))
    fig7.update_layout(title="Feature Correlation Matrix",
                       **layout(xaxis=dict(tickangle=-45, gridcolor="#1e2235"),
                                yaxis=dict(autorange="reversed", gridcolor="#1e2235"),
                                height=520))
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("#### 🏆 Top Correlations with Quality Score")
    cq     = corr["quality"].drop("quality").sort_values(key=abs, ascending=False)
    cbars  = [C["a1"] if v >= 0 else C["a2"] for v in cq.values]
    fig8   = go.Figure(go.Bar(x=cq.index, y=cq.values, marker_color=cbars,
                              text=cq.values.round(3), textposition="outside",
                              textfont=dict(color=C["text"], size=10)))
    fig8.update_layout(title="Correlation with Quality Score",
                       **layout(xaxis_title="Feature", yaxis_title="Pearson r",
                                yaxis=dict(range=[-1, 1], gridcolor="#1e2235")))
    st.plotly_chart(fig8, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – FEATURE EXPLORER
# ══════════════════════════════════════════════
with tab4:
    col4a, col4b = st.columns([3, 1])

    with col4a:
        if color_by_quality:
            cvals, cname, cscale = df["quality"], "Quality", "Viridis"
        else:
            cvals = df["color"].map({"red": 0, "white": 1})
            cname, cscale = "Color (0=red,1=white)", [[0, C["red"]], [1, C["white"]]]

        fig9 = go.Figure(go.Scatter(
            x=df[x_feat], y=df[y_feat], mode="markers",
            marker=dict(color=cvals, colorscale=cscale, size=5, opacity=0.65,
                        colorbar=dict(title=cname, tickfont=dict(color=C["text"])),
                        showscale=True),
            text=[f"Quality: {q}<br>Wine: {c}" for q, c in zip(df["quality"], df["color"])],
            hovertemplate=f"{x_feat}: %{{x:.3f}}<br>{y_feat}: %{{y:.3f}}<br>%{{text}}<extra></extra>",
        ))
        fig9.update_layout(title=f"{x_feat}  vs  {y_feat}",
                           **layout(xaxis_title=x_feat, yaxis_title=y_feat, height=460))
        st.plotly_chart(fig9, use_container_width=True)

    with col4b:
        for feat_label, feat_col in [(f"📌 {x_feat}", x_feat), (f"📌 {y_feat}", y_feat)]:
            st.markdown(f"**{feat_label}**")
            for lbl, val in [
                ("Mean",     df[feat_col].mean()),
                ("Median",   df[feat_col].median()),
                ("Std Dev",  df[feat_col].std()),
                ("Min",      df[feat_col].min()),
                ("Max",      df[feat_col].max()),
                ("Skewness", df[feat_col].skew()),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.2rem 0;border-bottom:1px solid #1e2235;'>"
                    f"<span style='color:#6b73a0;font-size:0.78rem;'>{lbl}</span>"
                    f"<span style='color:#e8ecff;font-weight:500;font-size:0.78rem;'>{val:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<br/>", unsafe_allow_html=True)

    # ── Parallel Coordinates ──
    st.markdown("#### 🔀 Parallel Coordinates")
    sdf = df if len(df) <= 2000 else df.sample(2000, random_state=42)
    fig10 = go.Figure(go.Parcoords(
        line=dict(color=sdf["quality"], colorscale="Viridis", showscale=True,
                  colorbar=dict(title="Quality", tickfont=dict(color=C["text"]))),
        dimensions=[dict(label=c.replace(" ", "<br>"), values=sdf[c])
                    for c in NUMERIC_COLS + ["quality"]],
        labelangle=-15, labelside="bottom",
    ))
    fig10.update_layout(title="Parallel Coordinates (≤ 2 000 rows)",
                        **layout(xaxis=None, yaxis=None, height=440,
                                 margin=dict(l=60, r=40, t=56, b=80)))
    st.plotly_chart(fig10, use_container_width=True)

    # ── Radar ──
    st.markdown("#### 🕸️ Normalised Feature Radar by Quality Grade")
    mn, mx = df[NUMERIC_COLS].min(), df[NUMERIC_COLS].max()
    ndf = (df[NUMERIC_COLS] - mn) / (mx - mn + 1e-9)
    ndf["quality"] = df["quality"].values
    radar = ndf.groupby("quality").mean().reset_index()
    rcols = px.colors.sequential.Viridis
    rqs   = sorted(radar["quality"].unique())

    fig11 = go.Figure()
    for i, q in enumerate(rqs):
        row  = radar[radar["quality"] == q].iloc[0]
        vals = [row[c] for c in NUMERIC_COLS] + [row[NUMERIC_COLS[0]]]
        labs = NUMERIC_COLS + [NUMERIC_COLS[0]]
        cidx = int(i / max(len(rqs)-1, 1) * (len(rcols)-1))
        clr  = rcols[cidx]
        fig11.add_trace(go.Scatterpolar(
            r=vals, theta=labs, name=f"Quality {int(q)}",
            line=dict(color=clr, width=2), fill="toself",
            fillcolor=clr.replace("rgb","rgba").replace(")",",.08)") if "rgb" in clr else clr,
        ))
    fig11.update_layout(title="Normalised Feature Profile by Quality",
                        **layout(xaxis=None, yaxis=None, height=460,
                                 polar=dict(
                                     bgcolor=C["card"],
                                     radialaxis=dict(gridcolor="#2a2e4a", tickfont=dict(color=C["muted"])),
                                     angularaxis=dict(gridcolor="#2a2e4a", tickfont=dict(color=C["muted"])),
                                 )))
    st.plotly_chart(fig11, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 – DATA TABLE
# ══════════════════════════════════════════════
with tab5:
    st.markdown(f"Displaying **{total:,}** rows")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=500)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Filtered CSV", csv, "wine_filtered.csv", "text/csv")
