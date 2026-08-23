"""Shared visual theme: light, blush/dusty-rose palette, and small HTML
component helpers so all four pages look consistent."""

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
COLOR_BG = "#FBF8F6"
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_ALT = "#F6EFEF"
COLOR_BORDER = "#E8DFDC"
COLOR_TEXT = "#33302E"
COLOR_TEXT_MUTED = "#7A716D"
COLOR_ACCENT = "#C88B96"          # dusty rose
COLOR_ACCENT_DARK = "#A66B77"
COLOR_ACCENT_SOFT = "#F1DEE2"     # very light pink
COLOR_POSITIVE = "#7A9E7E"        # muted sage green
COLOR_NEUTRAL = "#D9B36C"         # soft warm gold
COLOR_NEGATIVE = "#C08789"        # dusty red

SENTIMENT_COLORS = {
    "Positive": COLOR_POSITIVE,
    "Neutral": COLOR_NEUTRAL,
    "Negative": COLOR_NEGATIVE,
}

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLORWAY = [COLOR_ACCENT, COLOR_NEUTRAL, COLOR_POSITIVE, COLOR_NEGATIVE, "#9BB0C1", "#B7A6C9"]


def inject_global_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLOR_BG};
        }}
        html, body, [class*="css"] {{
            color: {COLOR_TEXT};
        }}
        #MainMenu, footer, header {{visibility: hidden;}}
        div.block-container {{
            padding-top: 1.2rem;
            max-width: 1180px;
        }}

        /* ---- Top navigation (streamlit-option-menu) ---- */
        .nav-wrapper {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 6px 10px;
            margin-bottom: 28px;
            box-shadow: 0 1px 3px rgba(60, 40, 40, 0.04);
        }}

        /* ---- Generic surface card ---- */
        .app-card {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 1px 4px rgba(60, 40, 40, 0.05);
            margin-bottom: 18px;
        }}

        /* ---- KPI card ---- */
        .kpi-card {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-left: 4px solid {COLOR_ACCENT};
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 4px rgba(60, 40, 40, 0.05);
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: {COLOR_TEXT_MUTED};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            line-height: 1.15;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {COLOR_TEXT_MUTED};
            margin-top: 4px;
        }}

        /* ---- Section header ---- */
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {COLOR_TEXT};
            margin: 6px 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid {COLOR_BORDER};
        }}
        .page-title {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {COLOR_TEXT};
            margin-bottom: 2px;
        }}
        .page-subtitle {{
            font-size: 0.95rem;
            color: {COLOR_TEXT_MUTED};
            margin-bottom: 22px;
        }}

        /* ---- Verdict banner ---- */
        .verdict-banner {{
            border-radius: 14px;
            padding: 22px 26px;
            text-align: center;
            border: 1px solid {COLOR_BORDER};
            margin: 10px 0 18px 0;
        }}
        .verdict-positive {{
            background-color: #EEF3EE;
            border-left: 6px solid {COLOR_POSITIVE};
        }}
        .verdict-negative {{
            background-color: #F5EBEC;
            border-left: 6px solid {COLOR_NEGATIVE};
        }}
        .verdict-neutral {{
            background-color: #FBF3E3;
            border-left: 6px solid {COLOR_NEUTRAL};
        }}
        .verdict-label {{
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {COLOR_TEXT_MUTED};
        }}
        .verdict-value {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 6px 0;
        }}
        .verdict-conf {{
            font-size: 0.95rem;
            color: {COLOR_TEXT_MUTED};
        }}

        /* ---- Pills / badges ---- */
        .pill {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background-color: {COLOR_ACCENT_SOFT};
            color: {COLOR_ACCENT_DARK};
        }}

        /* ---- Inputs ---- */
        .stTextArea textarea, .stTextInput input, .stNumberInput input {{
            border-radius: 10px !important;
            border: 1px solid {COLOR_BORDER} !important;
        }}
        .stButton > button {{
            background-color: {COLOR_ACCENT};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background-color: {COLOR_ACCENT_DARK};
            color: white;
        }}
        .stDownloadButton > button {{
            background-color: {COLOR_SURFACE};
            color: {COLOR_ACCENT_DARK};
            border: 1px solid {COLOR_ACCENT};
            border-radius: 10px;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_banner(label: str, sentiment_kind: str, sub_text: str = ""):
    """sentiment_kind in {'positive', 'negative', 'neutral'} controls styling."""
    css_class = {
        "positive": "verdict-positive",
        "negative": "verdict-negative",
        "neutral": "verdict-neutral",
    }.get(sentiment_kind, "verdict-neutral")
    sub_html = f'<div class="verdict-conf">{sub_text}</div>' if sub_text else ""
    st.markdown(
        f"""
        <div class="verdict-banner {css_class}">
            <div class="verdict-label">Result</div>
            <div class="verdict-value">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
