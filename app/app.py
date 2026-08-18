# ============================================================
# TRUTHLENS
# AI-POWERED FAKE NEWS DETECTOR
# ============================================================

import re
from pathlib import Path
from urllib.parse import quote

import joblib
import requests
import streamlit as st

from bs4 import BeautifulSoup


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "model_e.pkl"
TFIDF_PATH = APP_DIR / "model_e_tfidf.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found:\n\n{MODEL_PATH}"
        )
        st.stop()

    if not TFIDF_PATH.exists():
        st.error(
            f"TF-IDF vectorizer not found:\n\n{TFIDF_PATH}"
        )
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)

    except Exception as error:
        st.error(
            "Could not load the saved model files."
        )
        st.exception(error)
        st.stop()

    return model, tfidf


model, tfidf = load_model()


# ============================================================
# PREPROCESSING
# ============================================================

@st.cache_resource
def load_nlp():

    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        try:
            stop_words = set(
                stopwords.words("english")
            )

        except LookupError:

            import nltk

            nltk.download(
                "stopwords",
                quiet=True
            )

            stop_words = set(
                stopwords.words("english")
            )

        lemmatizer = WordNetLemmatizer()

        try:
            lemmatizer.lemmatize("test")

        except LookupError:

            import nltk

            nltk.download(
                "wordnet",
                quiet=True
            )

            nltk.download(
                "omw-1.4",
                quiet=True
            )

            lemmatizer = WordNetLemmatizer()

        return stop_words, lemmatizer

    except Exception as error:

        st.error(
            "Could not initialize NLP preprocessing."
        )

        st.exception(error)

        st.stop()


stop_words, lemmatizer = load_nlp()


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def lemmatize_text(text):

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def preprocess_text(text):

    text = clean_text(text)

    text = remove_stopwords(text)

    text = lemmatize_text(text)

    return text


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

def predict_news(article):

    processed_text = preprocess_text(article)

    if not processed_text:
        return None

    vectorized_text = tfidf.transform(
        [processed_text]
    )

    prediction = model.predict(
        vectorized_text
    )[0]

    if prediction == 1:
        return "REAL NEWS"

    return "FAKE NEWS"


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(
    headline,
    article_text,
    max_results=5
):

    # Use headline as the primary query.
    # This gives much cleaner news-search results
    # than sending the entire article.

    query_text = headline.strip()

    if not query_text:

        query_text = article_text[:200]

    encoded_query = quote(query_text)

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            rss_url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        # We intentionally use the HTML parser.
        # It can still read the RSS XML structure for
        # our simple <item>, <title>, <link>, <pubDate>
        # extraction and does not require lxml.

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        results = []

        for item in soup.find_all("item"):

            if len(results) >= max_results:
                break

            title_tag = item.find("title")
            link_tag = item.find("link")
            date_tag = item.find("pubDate")

            title = ""

            if title_tag:
                title = title_tag.get_text(
                    " ",
                    strip=True
                )

            link = ""

            if link_tag:

                link = link_tag.get_text(
                    " ",
                    strip=True
                )

                # Some RSS parsers expose the link
                # differently.

                if not link:
                    link = link_tag.string or ""

            date = ""

            if date_tag:

                date = date_tag.get_text(
                    " ",
                    strip=True
                )

            if title:

                results.append(
                    {
                        "title": title,
                        "link": link,
                        "date": date,
                    }
                )

        return results

    except requests.RequestException:
        return []

    except Exception:
        return []


# ============================================================
# WEB EVIDENCE
# ============================================================

def verify_web(
    headline,
    article_text
):

    sources = search_web(
        headline,
        article_text,
        max_results=5
    )

    evidence_score = len(sources)

    if evidence_score >= 4:

        status = "STRONG WEB EVIDENCE"

    elif evidence_score >= 2:

        status = "MODERATE WEB EVIDENCE"

    elif evidence_score == 1:

        status = "WEAK WEB EVIDENCE"

    else:

        status = "NO WEB EVIDENCE"


    return {
        "status": status,
        "score": evidence_score,
        "sources": sources,
    }


# ============================================================
# FINAL DECISION
# ============================================================

def analyze_article(
    headline,
    article_text
):

    ml_prediction = predict_news(
        article_text
    )

    web_result = verify_web(
        headline,
        article_text
    )

    web_score = web_result["score"]


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if web_score >= 4:

        if ml_prediction == "REAL NEWS":

            final_result = (
                "REAL — STRONG WEB SUPPORT"
            )

            result_type = "success"

        else:

            final_result = (
                "MODEL/WEB CONFLICT — REVIEW REQUIRED"
            )

            result_type = "warning"


    elif web_score >= 2:

        if ml_prediction == "REAL NEWS":

            final_result = (
                "REAL — WEB EVIDENCE SUPPORTS MODEL"
            )

            result_type = "success"

        else:

            final_result = (
                "MODEL/WEB CONFLICT — REVIEW REQUIRED"
            )

            result_type = "warning"


    else:

        if ml_prediction == "REAL NEWS":

            final_result = (
                "REAL — LIMITED WEB EVIDENCE"
            )

            result_type = "info"

        else:

            final_result = (
                "FAKE NEWS — LIMITED WEB EVIDENCE"
            )

            result_type = "error"


    return {
        "ml_prediction": ml_prediction,
        "web_status": web_result["status"],
        "web_score": web_score,
        "final_result": final_result,
        "result_type": result_type,
        "sources": web_result["sources"],
    }


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ TruthLens")

st.subheader(
    "AI-Powered Fake News Detection & Web Verification"
)

st.write(
    "Analyze a news article using your trained "
    "machine-learning model and compare the prediction "
    "with current web evidence."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📰 Check a News Article")

headline = st.text_input(
    "News Headline",
    placeholder=(
        "Example: RBI intervenes across markets "
        "to steady the rupee"
    ),
)

article_text = st.text_area(
    "Article Text",
    placeholder=(
        "Paste the complete article text here..."
    ),
    height=280,
)


verify_button = st.button(
    "🔍 Verify Article",
    type="primary",
    use_container_width=True,
)


# ============================================================
# VALIDATION + ANALYSIS
# ============================================================

if verify_button:

    if not headline.strip():

        st.warning(
            "Please enter the news headline."
        )

        st.stop()


    if not article_text.strip():

        st.warning(
            "Please paste the article text."
        )

        st.stop()


    with st.spinner(
        "Analyzing the article and checking web evidence..."
    ):

        result = analyze_article(
            headline,
            article_text
        )


    st.divider()


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("📊 Verification Result")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "ML Prediction",
            result["ml_prediction"]
        )


    with col2:

        st.metric(
            "Web Evidence",
            result["web_status"]
        )


    with col3:

        st.metric(
            "Evidence Score",
            f'{result["web_score"]} / 5'
        )


    st.divider()


    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.header("🎯 Final Assessment")


    if result["result_type"] == "success":

        st.success(
            result["final_result"]
        )


    elif result["result_type"] == "warning":

        st.warning(
            result["final_result"]
        )


    elif result["result_type"] == "error":

        st.error(
            result["final_result"]
        )


    else:

        st.info(
            result["final_result"]
        )


    # ========================================================
    # WEB SOURCES
    # ========================================================

    st.header("🌐 Supporting Web Reports")


    if not result["sources"]:

        st.info(
            "No matching news reports were found "
            "for this headline."
        )

    else:

        for index, source in enumerate(
            result["sources"],
            start=1
        ):

            st.markdown(
                f"**{index}. {source['title']}**"
            )

            if source["date"]:

                st.caption(
                    source["date"]
                )

            if source["link"]:

                st.link_button(
                    "Open Source",
                    source["link"]
                )

            st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ About TruthLens")

    st.write(
        "TruthLens combines machine learning with "
        "current web evidence to help assess news articles."
    )

    st.divider()

    st.write(
        "**Machine Learning**"
    )

    st.write(
        "Model E + TF-IDF"
    )

    st.write(
        "**Web Verification**"
    )

    st.write(
        "Current news search results"
    )

    st.divider()

    st.caption(
        "Web evidence is a verification signal, "
        "not absolute proof of truth."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TruthLens • Fake News Detection & Web Evidence Verification"
)