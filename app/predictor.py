import joblib
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_DIR / "linear_svm_model.pkl")

# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def predict_news(article):
    # Preprocess the article
    text = clean_text(article)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    # Convert text to TF-IDF
    article_tfidf = tfidf.transform([text])

    # Predict
    prediction = model.predict(article_tfidf)[0]

    if prediction == 0:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"
    