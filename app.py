"""
Women's Clothing E-Commerce Review Intelligence
================================================
Streamlit application built on top of the existing, already-trained NLP
pipeline (TF-IDF vectorizer + best sentiment model) and the rule-based
Product Recommendation system defined in the source notebook.

No model is trained or refit anywhere in this app -- only `.transform()` and
`.predict()` are ever called on the existing fitted objects, loaded from
Hugging Face.
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from data_loader import (
    analyze_product,
    load_dataset,
    load_model,
    load_vectorizer,
    predict_batch,
    predict_single_review,
)
from nlp_pipeline import CLASS_ORDER, preprocess_review
from pdf_utils import extract_text_from_pdf, split_into_reviews
from styles import (
    COLOR_ACCENT,
    COLOR_ACCENT_DARK,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    PLOTLY_COLORWAY,
    PLOTLY_TEMPLATE,
    SENTIMENT_COLORS,
    inject_global_css,
    kpi_card,
    page_header,
    section_title,
    verdict_banner,
)

st.set_page_config(
    page_title="Women's Clothing Review Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_css()

PAGES = ["Dataset Dashboard", "Review Prediction", "Product Recommendation", "PDF Analysis"]


# ---------------------------------------------------------------------------
# Shared chart formatting
# ---------------------------------------------------------------------------
def style_fig(fig, height=380, show_legend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        colorway=PLOTLY_COLORWAY,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(color=COLOR_TEXT, size=12.5),
        title_font=dict(size=14, color=COLOR_TEXT),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentiment_bar(pct_series, title):
    fig = px.bar(
        x=CLASS_ORDER,
        y=[pct_series.get(c, 0) for c in CLASS_ORDER],
        color=CLASS_ORDER,
        color_discrete_map=SENTIMENT_COLORS,
        text=[f"{pct_series.get(c, 0):.1f}%" for c in CLASS_ORDER],
        labels={"x": "Sentiment", "y": "Percentage"},
        title=title,
    )
    fig.update_traces(textposition="outside")
    return style_fig(fig, show_legend=False)


def sentiment_pie(pct_series, title):
    fig = px.pie(
        names=CLASS_ORDER,
        values=[pct_series.get(c, 0) for c in CLASS_ORDER],
        color=CLASS_ORDER,
        color_discrete_map=SENTIMENT_COLORS,
        title=title,
        hole=0.45,
    )
    return style_fig(fig, show_legend=True)


# ---------------------------------------------------------------------------
# Top navigation
# ---------------------------------------------------------------------------
def top_navigation():
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=PAGES,
        icons=[" " for _ in PAGES],
        orientation="horizontal",
        default_index=PAGES.index(st.session_state.page),
        styles={
            "container": {"padding": "2px 4px", "background-color": "transparent"},
            "icon": {"display": "none", "font-size": "0px"},
            "nav-link": {
                "font-size": "0.95rem",
                "font-weight": "600",
                "text-align": "center",
                "margin": "2px",
                "padding": "12px 16px",
                "color": COLOR_TEXT_MUTED,
                "border-radius": "10px",
                "background-color": "transparent",
            },
            "nav-link-selected": {
                "background-color": COLOR_ACCENT,
                "color": "white",
                "font-weight": "700",
            },
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.page = selected
    return selected


# ---------------------------------------------------------------------------
# Page 1 -- Dataset Dashboard
# ---------------------------------------------------------------------------
def dashboard_page(df: pd.DataFrame):
    page_header(
        "Dataset Dashboard",
        "Exploratory overview of the Women's Clothing E-Commerce Reviews dataset.",
    )

    section_title("Filters")
    with st.container():
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            departments = sorted(df["Department Name"].dropna().unique().tolist())
            selected_departments = st.multiselect("Department", departments, default=departments)
        with f2:
            classes = sorted(df["Class Name"].dropna().unique().tolist())
            selected_classes = st.multiselect("Class", classes, default=classes)
        with f3:
            rec_filter = st.selectbox("Recommendation Status", ["All", "Recommended", "Not Recommended"])

        f4, f5 = st.columns(2)
        with f4:
            ratings = sorted(df["Rating"].dropna().unique().tolist())
            selected_ratings = st.multiselect("Rating", ratings, default=ratings)
        with f5:
            age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
            age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = df[
        df["Department Name"].isin(selected_departments)
        & df["Class Name"].isin(selected_classes)
        & df["Rating"].isin(selected_ratings)
        & df["Age"].between(age_range[0], age_range[1])
    ]
    if rec_filter == "Recommended":
        filtered = filtered[filtered["Recommended IND"] == 1]
    elif rec_filter == "Not Recommended":
        filtered = filtered[filtered["Recommended IND"] == 0]

    if filtered.empty:
        st.warning("No reviews match the selected filters. Adjust the filters above.")
        return

    section_title("Key Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Total Reviews", f"{len(filtered):,}")
    with k2:
        kpi_card("Products", f"{filtered['Clothing ID'].nunique():,}")
    with k3:
        kpi_card("Average Rating", f"{filtered['Rating'].mean():.2f} / 5")
    with k4:
        kpi_card("Recommendation Rate", f"{filtered['Recommended IND'].mean() * 100:.1f}%")
    with k5:
        kpi_card("Average Age", f"{filtered['Age'].mean():.1f} yrs")
    with k6:
        kpi_card("Avg. Positive Feedback", f"{filtered['Positive Feedback Count'].mean():.1f}")

    section_title("Ratings & Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        rating_counts = filtered["Rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index.astype(str), y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"}, title="Rating Distribution",
        )
        fig.update_traces(marker_color=COLOR_ACCENT)
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
    with c2:
        rec_counts = filtered["Recommended IND"].map({1: "Recommended", 0: "Not Recommended"}).value_counts()
        fig = px.pie(
            names=rec_counts.index, values=rec_counts.values, hole=0.45,
            color=rec_counts.index,
            color_discrete_map={"Recommended": SENTIMENT_COLORS["Positive"], "Not Recommended": SENTIMENT_COLORS["Negative"]},
            title="Recommended vs Not Recommended",
        )
        st.plotly_chart(style_fig(fig), width="stretch")

    section_title("Sentiment Overview")
    c3, c4 = st.columns(2)
    sentiment_counts = filtered["Sentiment"].value_counts().reindex(CLASS_ORDER).fillna(0)
    sentiment_pct = (sentiment_counts / sentiment_counts.sum() * 100)
    with c3:
        st.plotly_chart(sentiment_bar(sentiment_pct, "Sentiment Distribution (%)"), width="stretch")
    with c4:
        st.plotly_chart(sentiment_pie(sentiment_pct, "Sentiment Share"), width="stretch")

    section_title("Reviews by Product Category")
    c5, c6 = st.columns(2)
    with c5:
        dept_counts = filtered["Department Name"].value_counts()
        fig = px.bar(
            x=dept_counts.values, y=dept_counts.index, orientation="h",
            labels={"x": "Reviews", "y": "Department"}, title="Reviews by Department",
        )
        fig.update_traces(marker_color=COLOR_ACCENT)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
    with c6:
        class_counts = filtered["Class Name"].value_counts().head(15)
        fig = px.bar(
            x=class_counts.values, y=class_counts.index, orientation="h",
            labels={"x": "Reviews", "y": "Class"}, title="Reviews by Class (Top 15)",
        )
        fig.update_traces(marker_color=COLOR_ACCENT_DARK)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, show_legend=False, height=440), width="stretch")

    section_title("Customer Demographics & Engagement")
    c7, c8 = st.columns(2)
    with c7:
        fig = px.histogram(filtered, x="Age", nbins=30, title="Age Distribution")
        fig.update_traces(marker_color=COLOR_ACCENT)
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
    with c8:
        fig = px.histogram(filtered, x="Positive Feedback Count", nbins=30, title="Positive Feedback Count Distribution")
        fig.update_traces(marker_color=COLOR_ACCENT_DARK)
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

    section_title("Department-Level Breakdown")
    c9, c10 = st.columns(2)
    with c9:
        avg_rating_dept = filtered.groupby("Department Name")["Rating"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=avg_rating_dept.values, y=avg_rating_dept.index, orientation="h",
            labels={"x": "Average Rating", "y": "Department"}, title="Average Rating by Department",
        )
        fig.update_traces(marker_color=COLOR_ACCENT)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
    with c10:
        rec_rate_dept = (filtered.groupby("Department Name")["Recommended IND"].mean() * 100).sort_values(ascending=False)
        fig = px.bar(
            x=rec_rate_dept.values, y=rec_rate_dept.index, orientation="h",
            labels={"x": "Recommendation Rate (%)", "y": "Department"}, title="Recommendation Rate by Department",
        )
        fig.update_traces(marker_color=COLOR_ACCENT_DARK)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")

    section_title("Sentiment by Department")
    dept_sentiment = (
        pd.crosstab(filtered["Department Name"], filtered["Sentiment"], normalize="index")
        .reindex(columns=CLASS_ORDER, fill_value=0) * 100
    )
    fig = go.Figure()
    for cls in CLASS_ORDER:
        fig.add_bar(name=cls, x=dept_sentiment.index, y=dept_sentiment[cls], marker_color=SENTIMENT_COLORS[cls])
    fig.update_layout(barmode="stack", title="Sentiment Distribution by Department (%)", xaxis_title="Department", yaxis_title="Percentage")
    st.plotly_chart(style_fig(fig, height=420), width="stretch")

    with st.expander("View a sample of the filtered data"):
        preview_cols = [
            "Clothing ID", "Age", "Review Text", "Rating", "Recommended IND",
            "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "Sentiment",
        ]
        preview_cols = [c for c in preview_cols if c in filtered.columns]
        st.dataframe(filtered[preview_cols].head(200), width="stretch")


# ---------------------------------------------------------------------------
# Page 2 -- Review Prediction
# ---------------------------------------------------------------------------
def prediction_page(vectorizer, model):
    page_header(
        "Review Prediction",
        "Enter a new customer review and get its predicted sentiment from the trained model.",
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    review_text = st.text_area(
        "Customer review",
        placeholder="I absolutely love this dress. It fits perfectly and the material is amazing.",
        height=140,
    )
    run = st.button("Analyze Sentiment")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        return

    if not review_text or not review_text.strip():
        st.warning("Please enter a review before analyzing.")
        return

    label, proba_dict, proba_kind = predict_single_review(review_text, vectorizer, model)

    sentiment_kind = label.lower()
    confidence_text = ""
    if proba_dict is not None:
        confidence = proba_dict.get(label, 0) * 100
        if proba_kind == "predict_proba":
            confidence_text = f"Confidence: {confidence:.1f}%"
        else:
            confidence_text = f"Approximate confidence (from decision scores): {confidence:.1f}%"

    verdict_banner(label.upper(), sentiment_kind, confidence_text)

    if proba_dict is not None:
        section_title("Class Probabilities")
        pct_series = pd.Series({c: proba_dict.get(c, 0) * 100 for c in CLASS_ORDER})
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.plotly_chart(sentiment_bar(pct_series, "Predicted Probability by Class"), width="stretch")
        with c2:
            for c in CLASS_ORDER:
                kpi_card(f"{c} Probability", f"{pct_series[c]:.1f}%")
    else:
        st.info("The loaded model does not expose class probabilities or decision scores -- only the final predicted label is available.")


# ---------------------------------------------------------------------------
# Page 3 -- Product Recommendation
# ---------------------------------------------------------------------------
def recommendation_page(df: pd.DataFrame, vectorizer, model):
    page_header(
        "Product Recommendation",
        "Look up a Clothing ID to get a transparent, rule-based Recommended / Not Recommended verdict.",
    )

    default_id = int(df["Clothing ID"].value_counts().idxmax())

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        clothing_id = st.number_input(
            "Clothing ID",
            min_value=0,
            value=default_id,
            step=1,
            help=f"Example: the most-reviewed product in the dataset is Clothing ID {default_id}.",
        )
    with col_b:
        st.write("")
        st.write("")
        search = st.button("Search Product")
    st.markdown("</div>", unsafe_allow_html=True)

    if not search:
        return

    result = analyze_product(int(clothing_id), df, vectorizer, model)

    if result is None:
        st.warning(f"No reviews found for Clothing ID {int(clothing_id)}. Please check the ID and try again.")
        return

    section_title("Product Statistics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Clothing ID", f"{result['clothing_id']}")
    with k2:
        kpi_card("Number of Reviews", f"{result['n_reviews']:,}")
    with k3:
        kpi_card("Average Rating", f"{result['avg_rating']:.2f} / 5")
    with k4:
        kpi_card("Recommendation Rate", f"{result['rec_rate']:.1f}%")

    k5, k6, k7 = st.columns(3)
    with k5:
        kpi_card("Positive", f"{result['positive_pct']:.1f}%")
    with k6:
        kpi_card("Neutral", f"{result['neutral_pct']:.1f}%")
    with k7:
        kpi_card("Negative", f"{result['negative_pct']:.1f}%")

    section_title("Product Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        rating_counts = result["product_df"]["Rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index.astype(str), y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"}, title="Rating Distribution",
        )
        fig.update_traces(marker_color=COLOR_ACCENT)
        st.plotly_chart(style_fig(fig, show_legend=False), width="stretch")
    with c2:
        pct_series = pd.Series({
            "Positive": result["positive_pct"],
            "Neutral": result["neutral_pct"],
            "Negative": result["negative_pct"],
        })
        st.plotly_chart(sentiment_pie(pct_series, "Predicted Sentiment Mix"), width="stretch")
    with c3:
        rec_pie = pd.Series({"Recommended": result["rec_rate"], "Not Recommended": result["not_rec_rate"]})
        fig = px.pie(
            names=rec_pie.index, values=rec_pie.values, hole=0.45,
            color=rec_pie.index,
            color_discrete_map={"Recommended": SENTIMENT_COLORS["Positive"], "Not Recommended": SENTIMENT_COLORS["Negative"]},
            title="Historical Recommendation",
        )
        st.plotly_chart(style_fig(fig), width="stretch")

    section_title("Final Recommendation")
    is_rec = result["final_label"] == "Recommended"
    verdict_banner(
        result["final_label"].upper(),
        "positive" if is_rec else "negative",
        "Rule: Average Rating ≥ 4, Recommendation Rate ≥ 70%, and Positive % > Negative %",
    )


# ---------------------------------------------------------------------------
# Page 4 -- PDF Analysis
# ---------------------------------------------------------------------------
def pdf_page(vectorizer, model):
    page_header(
        "PDF Analysis",
        "Upload a PDF containing multiple customer reviews for batch sentiment analysis.",
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        return

    with st.spinner("Extracting text from PDF..."):
        try:
            full_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
        except Exception as exc:
            st.error(f"Could not read this PDF file: {exc}")
            return

    if not full_text or not full_text.strip():
        st.warning("No extractable text was found in this PDF. It may be a scanned/image-based document.")
        return

    raw_reviews = split_into_reviews(full_text)
    if not raw_reviews:
        st.warning("Could not identify individual reviews in this PDF's structure. Try a differently formatted file.")
        return

    with st.spinner(f"Analyzing {len(raw_reviews)} reviews..."):
        processed_reviews = [preprocess_review(r) for r in raw_reviews]
        predictions = predict_batch(processed_reviews, vectorizer, model)

    results_df = pd.DataFrame({
        "Review #": range(1, len(raw_reviews) + 1),
        "Review Text": raw_reviews,
        "Predicted Sentiment": predictions,
    })

    total = len(results_df)
    sentiment_counts = results_df["Predicted Sentiment"].value_counts().reindex(CLASS_ORDER).fillna(0)
    sentiment_pct = (sentiment_counts / total * 100)

    section_title("Batch Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Reviews Analyzed", f"{total:,}")
    with k2:
        kpi_card("Positive", f"{sentiment_pct['Positive']:.1f}%")
    with k3:
        kpi_card("Neutral", f"{sentiment_pct['Neutral']:.1f}%")
    with k4:
        kpi_card("Negative", f"{sentiment_pct['Negative']:.1f}%")

    section_title("Sentiment Distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(sentiment_bar(sentiment_pct, "Sentiment Distribution (%)"), width="stretch")
    with c2:
        st.plotly_chart(sentiment_pie(sentiment_pct, "Sentiment Share"), width="stretch")

    section_title("Review-Level Results")
    filter_choice = st.multiselect("Filter by sentiment", CLASS_ORDER, default=CLASS_ORDER)
    filtered_results = results_df[results_df["Predicted Sentiment"].isin(filter_choice)]
    st.dataframe(filtered_results, width="stretch", height=380)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Results as CSV",
        data=csv_bytes,
        file_name="pdf_review_sentiment_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    selected_page = top_navigation()

    try:
        model = load_model()
        vectorizer = load_vectorizer()
        df = load_dataset()
    except Exception as exc:
        st.error(
            "Could not load one or more required resources (model, vectorizer, or dataset) "
            f"from Hugging Face. Details: {exc}"
        )
        st.stop()

    if selected_page == "Dataset Dashboard":
        dashboard_page(df)
    elif selected_page == "Review Prediction":
        prediction_page(vectorizer, model)
    elif selected_page == "Product Recommendation":
        recommendation_page(df, vectorizer, model)
    elif selected_page == "PDF Analysis":
        pdf_page(vectorizer, model)


if __name__ == "__main__":
    main()
