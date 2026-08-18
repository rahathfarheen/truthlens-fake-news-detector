# 🛡️ TruthLens — AI-Powered Fake News Detector

TruthLens is an AI-powered fake news detection system that combines
machine learning with current web evidence to help assess the
credibility of news articles.

The system uses a TF-IDF + Linear SVM machine-learning model and
supplements the prediction with current news search results.

---

## 🚀 Features

- 🧠 Machine-learning based fake news classification
- 📊 TF-IDF text representation
- ⚡ Linear SVM classifier
- 🌐 Current web evidence verification
- 🇮🇳 Testing with Indian news examples
- ⚠️ ML/Web conflict detection
- 🖥️ Clean Streamlit user interface
- 📰 Supporting source links
- 📈 Model evaluation with confusion matrix and classification report

---

## 🏗️ System Architecture

```text
                News Article
                     │
                     ▼
              Text Preprocessing
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   TF-IDF + ML              Web Verification
        │                         │
        ▼                         ▼
   ML Prediction             Web Evidence
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
              Final Assessment
                     │
                     ▼
               Streamlit UI


🤖 Machine Learning

The final model uses:

TF-IDF Vectorization
Unigrams and bigrams
Sublinear TF scaling
Linear Support Vector Machine
Model E Performance

Test accuracy:

99.21%

Confusion Matrix:

[[3450   41]
 [  20 4219]]

Classification performance:

Fake:
Precision: 0.99
Recall:    0.99
F1-score:  0.99


Real:
Precision: 0.99
Recall:    1.00
F1-score:  0.99
🌐 Web Verification

Machine-learning predictions can sometimes be wrong when dealing
with recent events.

TruthLens therefore performs a current news search and compares
the machine-learning prediction with available web evidence.

The system categorizes evidence as:

Strong Web Evidence
Moderate Web Evidence
Weak Web Evidence
No Web Evidence

If the ML prediction conflicts with strong current web evidence,
the application reports:

MODEL/WEB CONFLICT — REVIEW REQUIRED

This prevents the system from blindly treating the ML prediction
as absolute truth.

🇮🇳 Indian News Testing

The system was additionally evaluated using Indian news examples,
including:

Jharkhand recruitment examination protests
NEET UG counselling
Pimpri-Chinchwad civic complaints
RBI and rupee intervention
Milky Mist market debut

One important example demonstrated a disagreement between the ML
model and current web evidence.

🖥️ Application

The application is built using Streamlit.

Users can enter:

News headline
Article text

The system then displays:

ML prediction
Web evidence
Evidence score
Final assessment
Supporting news sources

📁 Project Structure
fake-news-detector/
│
├── app/
│   ├── app.py
│   ├── model_e.pkl
│   └── model_e_tfidf.pkl
│
├── data/
│
├── notebooks/
│   └── 05_indian_news_testing.ipynb
│
├── test_data/
│
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd fake-news-detector

Create a virtual environment:

python -m venv .venv313

Activate it on Windows:

.venv313\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Run the Application

From the project root:

streamlit run app/app.py

The application will open in your browser.

🧪 Example
Input

Headline:

RBI intervenes across markets to steady the rupee

The ML model may classify the article as:

FAKE NEWS

while current web evidence may show strong support for the event.

TruthLens therefore reports:

MODEL/WEB CONFLICT — REVIEW REQUIRED

This demonstrates the purpose of combining machine learning
with current external evidence.

⚠️ Limitations
Machine-learning predictions are not guaranteed to be correct.
Web search results are evidence rather than absolute proof.
Search results can change over time.
Satire and opinion articles can be difficult to classify.
Very short articles may provide insufficient information.
The system currently focuses primarily on English-language text.

🔮 Future Improvements

Potential future improvements include:

Transformer-based models such as BERT
Multilingual Indian-language support
Better source credibility scoring
Article similarity detection
Duplicate-news detection
Claim-level verification
More comprehensive fact-checking sources
News-source reputation analysis
Docker deployment
Cloud deployment

🎓 Project Objective

The objective of TruthLens is to demonstrate how machine learning
and current web information can be combined to create a more
useful news-verification system.

Rather than treating a machine-learning prediction as absolute,
the system provides additional evidence and highlights conflicts
that require human review.

📜 Disclaimer

TruthLens is an educational and research project.

Its predictions and web evidence should not be considered
definitive proof that a news article is true or false.

Users should verify important claims using authoritative sources.

👨‍💻 Author

Rahath Farheen

GitHub: https://github.com/rahathfarheen

LinkedIn: https://www.linkedin.com/in/rahath-farheen-764926398
