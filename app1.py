import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VinIQ · Wine Quality Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root & background */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], section[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #0d0f18 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12151f !important;
    border-right: 1px solid #1e2235;
}
section[data-testid="stSidebar"] * {
    color: #c8cce0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #8890b0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Select boxes, dropdowns */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: #1a1d2e !important;
    border: 1px solid #2a2e45 !important;
    border-radius: 8px !important;
    color: #e0e4f0 !important;
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

/* Horizontal rule */
hr { border-color: #1e2235 !important; }

/* Block container */
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }

/* DataFrames */
[data-testid="stDataFrame"] { background: #13162a !important; }
iframe { background: #13162a !important; }

/* Plot bg override via css vars */
:root {
    --plot-bgcolor: #0d0f18;
    --paper-bgcolor: #0d0f18;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f18; }
::-webkit-scrollbar-thumb { background: #2a2e4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4265; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winequality.csv")
    df = pd.read_csv(csv_path)
    return df

@st.cache_data
def remove_outliers_iqr(df):
    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    df_clean = df.copy()
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lb) & (df_clean[col] <= ub)]
    return df_clean

df_raw = load_data()
df_clean = remove_outliers_iqr(df_raw)

NUMERIC_COLS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

COLORS = {
    "red":     "#e05f5f",
    "white":   "#a78bfa",
    "premium": "#34d399",
    "avg":     "#fbbf24",
    "accent1": "#60a5fa",
    "accent2": "#f472b6",
    "bg":      "#0d0f18",
    "card":    "#13162a",
    "border":  "#2a2e4a",
    "text":    "#e8ecff",
    "muted":   "#6b73a0",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(family="Inter", color=COLORS["text"]),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS["border"],
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="#1e2235", zerolinecolor="#1e2235"),
    yaxis=dict(gridcolor="#1e2235", zerolinecolor="#1e2235"),
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <span style='font-size:2.2rem;'>🍷</span>
        <h2 style='color:#e8ecff; margin:0.3rem 0 0 0; font-size:1.1rem; font-weight:700;'>Wine Quality</h2>
        <p style='color:#6b73a0; font-size:0.72rem; margin:0;'>Analytics Dashboard</p>
    </div>
    <hr style='border-color:#1e2235; margin:0.8rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**📋 Filters**")

    wine_type = st.selectbox("Wine Type", ["All", "Red", "White"])
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=False)

    quality_range = st.slider(
        "Quality Score Range",
        min_value=int(df_raw["quality"].min()),
        max_value=int(df_raw["quality"].max()),
        value=(int(df_raw["quality"].min()), int(df_raw["quality"].max()))
    )

    alcohol_range = st.slider(
        "Alcohol % Range",
        min_value=float(df_raw["alcohol"].min()),
        max_value=float(df_raw["alcohol"].max()),
        value=(float(df_raw["alcohol"].min()), float(df_raw["alcohol"].max())),
        step=0.1
    )

    st.markdown("<hr style='border-color:#1e2235; margin:0.8rem 0;'/>", unsafe_allow_html=True)
    st.markdown("**🔍 Feature Explorer**")

    x_feat = st.selectbox("X-Axis Feature", NUMERIC_COLS, index=NUMERIC_COLS.index("alcohol"))
    y_feat = st.selectbox("Y-Axis Feature", NUMERIC_COLS, index=NUMERIC_COLS.index("volatile acidity"))

    st.markdown("<hr style='border-color:#1e2235; margin:0.8rem 0;'/>", unsafe_allow_html=True)

    color_by_quality = st.checkbox("Color Scatter by Quality", value=True)

    st.markdown("<hr style='border-color:#1e2235; margin:0.8rem 0;'/>", unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#6b73a0; font-size:0.7rem; text-align:center;'>
        Data: winequality.csv<br/>Prashant's Wine Analytics
    </p>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER DATA
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

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
outlier_badge = f" &nbsp;·&nbsp; <span style='background:#1e2235;padding:2px 8px;border-radius:4px;font-size:0.75rem;'>Outliers Removed</span>" if remove_outliers else ""
header_html = (
    f"<div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;'>"
    f"<span style='font-size:2.5rem;'>🍷</span>"
    f"<div>"
    f"<h1 style='color:#e8ecff;margin:0;font-size:1.7rem;font-weight:700;'>Wine Quality Dashboard</h1>"
    f"<p style='color:#6b73a0;margin:0;font-size:0.82rem;'>Showing <b style='color:#a78bfa'>{len(df):,}</b> samples"
    f" &nbsp;·&nbsp; <b style='color:{COLORS['red']}'>{len(df_red):,}</b> Red"
    f" &nbsp;·&nbsp; <b style='color:{COLORS['white']}'>{len(df_white):,}</b> White"
    f"{outlier_badge}</p>"
    f"</div></div>"
    f"<hr style='border-color:#1e2235;margin:0.4rem 0 1rem 0;'/>"
)
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
if len(df) == 0:
    st.warning("No data matches the current filters. Please adjust your selections.")
    st.stop()

total_samples  = len(df)
premium_wines  = len(df[df["quality"] >= 7])
avg_quality    = round(df["quality"].mean(), 3)
avg_alcohol    = round(df["alcohol"].mean(), 2)
premium_pct    = round(premium_wines / total_samples * 100, 1)
raw_total      = len(df_base)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Samples",    f"{total_samples:,}",    f"{total_samples - raw_total:+,} vs full")
with c2:
    st.metric("Premium (≥7)",     f"{premium_wines:,}",    f"{premium_pct}% share")
with c3:
    st.metric("Avg Quality Score", avg_quality,            f"Max {int(df['quality'].max())}")
with c4:
    st.metric("Avg Alcohol %",    avg_alcohol,             f"Δ {round(df['alcohol'].std(),2)} σ")
with c5:
    red_pct = round(len(df_red)/len(df)*100, 1) if len(df) > 0 else 0
    st.metric("Red vs White",     f"{red_pct}% Red",       f"{round(100-red_pct,1)}% White")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Distributions", "🔥 Correlations", "🔬 Feature Explorer", "📋 Data"
])

# ══════════════════════════════════════════════
# TAB 1 – OVERVIEW (candlestick-style + donut)
# ══════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 1])

    with col_a:
        # Quality Distribution – bar / band chart
        quality_counts = df.groupby(["quality", "color"]).size().reset_index(name="count")
        
        fig_band = go.Figure()
        
        for color_val, line_color, fill_color in [
            ("red",   COLORS["red"],   "rgba(224,95,95,0.3)"),
            ("white", COLORS["white"], "rgba(167,139,250,0.3)"),
        ]:
            sub = quality_counts[quality_counts["color"] == color_val].sort_values("quality")
            if len(sub):
                fig_band.add_trace(go.Bar(
                    x=sub["quality"],
                    y=sub["count"],
                    name=color_val.capitalize(),
                    marker_color=line_color,
                    opacity=0.85,
                    width=0.4,
                    offset=-0.22 if color_val == "red" else 0.22,
                ))
        
        fig_band.update_layout(
            title="Quality Score Distribution by Wine Type",
            **PLOTLY_LAYOUT,
            barmode="overlay",
            xaxis_title="Quality Score",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_band, use_container_width=True)

    with col_b:
        # Donut – Red vs White
        fig_donut = go.Figure(go.Pie(
            labels=["Red", "White"],
            values=[len(df_red), len(df_white)],
            hole=0.62,
            marker=dict(colors=[COLORS["red"], COLORS["white"]],
                        line=dict(color=COLORS["bg"], width=3)),
            textinfo="percent+label",
            textfont=dict(color=COLORS["text"], size=11),
            showlegend=False,
        ))
        fig_donut.update_layout(
            title="Color Split",
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
            margin=dict(l=10, r=10, t=50, b=10),
            annotations=[dict(
                text=f"<b>{total_samples:,}</b><br>wines",
                x=0.5, y=0.5, font_size=13, font_color=COLORS["text"],
                showarrow=False
            )],
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Quality by Wine Type – Violin ──
    fig_violin = go.Figure()
    for color_val, vc in [("red", COLORS["red"]), ("white", COLORS["white"])]:
        sub = df[df["color"] == color_val]
        if len(sub):
            fig_violin.add_trace(go.Violin(
                x=sub["color"].str.capitalize(),
                y=sub["quality"],
                name=color_val.capitalize(),
                line_color=vc,
                fillcolor=vc.replace(")", ",0.2)").replace("rgb", "rgba") if "rgb" in vc else vc,
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                pointpos=0,
                marker=dict(color=vc, opacity=0.4, size=3),
            ))
    fig_violin.update_layout(
        title="Quality Distribution (Violin)",
        **PLOTLY_LAYOUT,
        yaxis_title="Quality Score",
        violingap=0.3,
        violinmode="overlay",
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    # ── Avg feature by quality ──
    st.markdown("#### 📊 Average Feature Values by Quality Score")
    avg_by_quality = df.groupby("quality")[NUMERIC_COLS].mean().reset_index()

    fig_heat = go.Figure(go.Heatmap(
        z=avg_by_quality[NUMERIC_COLS].T.values,
        x=avg_by_quality["quality"].astype(str),
        y=NUMERIC_COLS,
        colorscale="Viridis",
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
        text=np.round(avg_by_quality[NUMERIC_COLS].T.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
    ))
    fig_heat.update_layout(
        title="Mean Feature Values per Quality Grade",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(title="Quality Score", gridcolor="#1e2235"),
        yaxis=dict(autorange="reversed", gridcolor="#1e2235"),
        height=380,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – DISTRIBUTIONS (histograms)
# ══════════════════════════════════════════════
with tab2:
    feat_hist = st.selectbox("Select feature to inspect", NUMERIC_COLS, key="hist_feat")
    nbins_val = st.slider("Number of bins", 10, 100, 40, key="nbins")

    col2a, col2b = st.columns(2)

    with col2a:
        # Overlaid histogram Red vs White
        fig_hist = go.Figure()
        for color_val, vc in [("red", COLORS["red"]), ("white", COLORS["white"])]:
            sub = df[df["color"] == color_val]
            if len(sub):
                fig_hist.add_trace(go.Histogram(
                    x=sub[feat_hist],
                    name=color_val.capitalize(),
                    marker_color=vc,
                    opacity=0.70,
                    nbinsx=nbins_val,
                ))
        fig_hist.update_layout(
            title=f"{feat_hist} – Distribution",
            barmode="overlay",
            **PLOTLY_LAYOUT,
            xaxis_title=feat_hist,
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2b:
        # Box plot per quality
        fig_box = go.Figure()
        color_sequence = px.colors.sequential.Viridis
        qualities = sorted(df["quality"].unique())
        for i, q in enumerate(qualities):
            sub = df[df["quality"] == q]
            cidx = int(i / max(len(qualities)-1, 1) * (len(color_sequence)-1))
            fig_box.add_trace(go.Box(
                y=sub[feat_hist],
                name=f"Q{int(q)}",
                marker_color=color_sequence[cidx],
                boxmean=True,
            ))
        fig_box.update_layout(
            title=f"{feat_hist} by Quality Score",
            **PLOTLY_LAYOUT,
            yaxis_title=feat_hist,
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Summary statistics table ──
    st.markdown("#### 📋 Descriptive Statistics")
    stats_df = df[NUMERIC_COLS + ["quality"]].describe().round(4)
    st.dataframe(
        stats_df.style.background_gradient(cmap="Blues", axis=1),
        use_container_width=True,
    )

# ══════════════════════════════════════════════
# TAB 3 – CORRELATIONS
# ══════════════════════════════════════════════
with tab3:
    corr_cols = NUMERIC_COLS + ["quality"]
    corr = df[corr_cols].corr().round(3)

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
    ))
    fig_corr.update_layout(
        title="Feature Correlation Matrix",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(tickangle=-45, gridcolor="#1e2235"),
        yaxis=dict(autorange="reversed", gridcolor="#1e2235"),
        height=520,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Top correlations with quality
    st.markdown("#### 🏆 Top Correlations with Quality Score")
    corr_quality = corr["quality"].drop("quality").sort_values(key=abs, ascending=False)
    
    colors_bar = [COLORS["accent1"] if v >= 0 else COLORS["accent2"] for v in corr_quality.values]
    fig_bar_corr = go.Figure(go.Bar(
        x=corr_quality.index,
        y=corr_quality.values,
        marker_color=colors_bar,
        text=corr_quality.values.round(3),
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=10),
    ))
    fig_bar_corr.update_layout(
        title="Correlation with Quality Score",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis",)},
        xaxis_title="Feature",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1, 1], gridcolor="#1e2235"),
    )
    st.plotly_chart(fig_bar_corr, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – FEATURE EXPLORER (scatter + parallel)
# ══════════════════════════════════════════════
with tab4:
    col4a, col4b = st.columns([3, 1])

    with col4a:
        if color_by_quality:
            color_vals = df["quality"]
            color_name = "quality"
            cscale = "Viridis"
        else:
            color_vals = df["color"].map({"red": 0, "white": 1})
            color_name = "color (0=red,1=white)"
            cscale = [[0, COLORS["red"]], [1, COLORS["white"]]]

        fig_scatter = go.Figure(go.Scatter(
            x=df[x_feat],
            y=df[y_feat],
            mode="markers",
            marker=dict(
                color=color_vals,
                colorscale=cscale,
                size=5,
                opacity=0.65,
                colorbar=dict(title=color_name, tickfont=dict(color=COLORS["text"])),
                showscale=True,
            ),
            text=[f"Quality: {q}<br>Wine: {c}" for q, c in zip(df["quality"], df["color"])],
            hovertemplate=f"{x_feat}: %{{x:.3f}}<br>{y_feat}: %{{y:.3f}}<br>%{{text}}<extra></extra>",
        ))
        fig_scatter.update_layout(
            title=f"{x_feat} vs {y_feat}",
            **PLOTLY_LAYOUT,
            xaxis_title=x_feat,
            yaxis_title=y_feat,
            height=480,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4b:
        # Feature stats for selected axis
        st.markdown(f"**{x_feat} Stats**")
        for label, val in [
            ("Mean", df[x_feat].mean()),
            ("Median", df[x_feat].median()),
            ("Std Dev", df[x_feat].std()),
            ("Min", df[x_feat].min()),
            ("Max", df[x_feat].max()),
            ("Skewness", df[x_feat].skew()),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.22rem 0;border-bottom:1px solid #1e2235;'>"
                f"<span style='color:#6b73a0;font-size:0.8rem;'>{label}</span>"
                f"<span style='color:#e8ecff;font-weight:500;font-size:0.8rem;'>{val:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"**{y_feat} Stats**")
        for label, val in [
            ("Mean", df[y_feat].mean()),
            ("Median", df[y_feat].median()),
            ("Std Dev", df[y_feat].std()),
            ("Min", df[y_feat].min()),
            ("Max", df[y_feat].max()),
            ("Skewness", df[y_feat].skew()),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.22rem 0;border-bottom:1px solid #1e2235;'>"
                f"<span style='color:#6b73a0;font-size:0.8rem;'>{label}</span>"
                f"<span style='color:#e8ecff;font-weight:500;font-size:0.8rem;'>{val:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Parallel coordinates ──
    st.markdown("#### 🔀 Parallel Coordinates – All Features")
    pc_cols = NUMERIC_COLS + ["quality"]
    # Sample if large
    sample_df = df if len(df) <= 2000 else df.sample(2000, random_state=42)

    fig_pc = go.Figure(go.Parcoords(
        line=dict(
            color=sample_df["quality"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Quality", tickfont=dict(color=COLORS["text"])),
        ),
        dimensions=[
            dict(label=col.replace(" ", "<br>"), values=sample_df[col])
            for col in pc_cols
        ],
        labelangle=-15,
        labelside="bottom",
    ))
    fig_pc.update_layout(
        title="Parallel Coordinates (up to 2 000 samples)",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        height=450,
        margin=dict(l=60, r=40, t=60, b=80),
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    # ── Radar chart per quality group ──
    st.markdown("#### 🕸️ Radar – Avg Normalised Features by Quality Grade")
    min_vals = df[NUMERIC_COLS].min()
    max_vals = df[NUMERIC_COLS].max()
    norm_df  = (df[NUMERIC_COLS] - min_vals) / (max_vals - min_vals + 1e-9)
    norm_df["quality"] = df["quality"]
    radar_data = norm_df.groupby("quality").mean().reset_index()

    radar_qualities = sorted(radar_data["quality"].unique())
    radar_colors    = px.colors.sequential.Viridis

    fig_radar = go.Figure()
    for i, q in enumerate(radar_qualities):
        row = radar_data[radar_data["quality"] == q].iloc[0]
        vals = [row[c] for c in NUMERIC_COLS] + [row[NUMERIC_COLS[0]]]
        labs = NUMERIC_COLS + [NUMERIC_COLS[0]]
        cidx = int(i / max(len(radar_qualities)-1, 1) * (len(radar_colors)-1))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=labs, name=f"Quality {int(q)}",
            line=dict(color=radar_colors[cidx], width=2),
            fill="toself",
            fillcolor=radar_colors[cidx].replace("rgb", "rgba").replace(")", ",0.08)") if "rgb" in radar_colors[cidx] else radar_colors[cidx],
        ))
    fig_radar.update_layout(
        title="Normalised Feature Profile by Quality",
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(gridcolor="#2a2e4a", tickfont=dict(color=COLORS["muted"])),
            angularaxis=dict(gridcolor="#2a2e4a", tickfont=dict(color=COLORS["muted"])),
        ),
        height=480,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 – DATA TABLE
# ══════════════════════════════════════════════
with tab5:
    st.markdown(f"Displaying **{len(df):,}** rows")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=500)
    
    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Filtered CSV",
        data=csv,
        file_name="wine_filtered.csv",
        mime="text/csv",
    )
