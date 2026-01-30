import streamlit as st
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# Download NLTK resources (first run only)
# -------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------------------------------------
# Small Sample Dataset (Hardcoded)
# -------------------------------------------------
data = [
    {
        "text": "Software Engineer reputed IT company python java development",
        "label": 0
    },
    {
        "text": "Data entry work from home pay registration fee earn money fast",
        "label": 1
    },
    {
        "text": "Machine learning engineer AI startup python data science",
        "label": 0
    },
    {
        "text": "Online typing job no experience instant payment after fee",
        "label": 1
    },
    {
        "text": "Web developer company full time html css javascript",
        "label": 0
    },
    {
        "text": "Easy part time job earn daily income no skills required",
        "label": 1
    }
]

df = pd.DataFrame(data)

# -------------------------------------------------
# Text Preprocessing
# -------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["label"]

# -------------------------------------------------
# TF-IDF Feature Extraction
# -------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

# -------------------------------------------------
# Logistic Regression Model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42
)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------------------------
# Streamlit User Interface
# -------------------------------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake or Real Job Posting Detection")
st.markdown("**Machine Learning + NLP using Logistic Regression**")

st.info(f"Model trained on small sample data | Accuracy: {accuracy:.2f}")

st.subheader("📥 Enter Job Posting Text")

user_input = st.text_area(
    "Paste or type a job posting description here",
    height=200
)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter job posting text.")
    else:
        processed = preprocess_text(user_input)
        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ This job posting is likely **FAKE**")
        else:
            st.success("✅ This job posting is likely **REAL**")

        st.write("Confidence Score:")
        st.progress(float(max(probability)))
