"""Machine-learning module: preprocessing, training, prediction."""
import os, re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.joblib")


def clean_text(text):
    """Lowercase, strip URLs/punctuation/digits, remove stop-words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(words)


def load_dataset(path):
    """CSV must have columns: text,label  (label = FAKE/REAL or 0/1)."""
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.upper().map(
        {"FAKE": 1, "REAL": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["clean"] = df["text"].apply(clean_text)
    return df


def train(csv_path):
    """Train 3 models, keep the best by F1, save it, return metrics dict."""
    df = load_dataset(csv_path)
    if len(df) < 10 or df["label"].nunique() < 2:
        raise ValueError("Need at least 10 rows containing both FAKE and REAL labels")
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    best = None
    comparison = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                         ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        m = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "confusion": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
        }
        comparison[name] = m
        if best is None or m["f1"] > best[2]["f1"]:
            best = (name, pipe, m)

    joblib.dump(best[1], MODEL_PATH)
    return {"model_name": best[0], "rows": len(df), **best[2], "comparison": comparison}


def model_exists():
    return os.path.exists(MODEL_PATH)


def predict(text):
    """Return (label, confidence%) where label is 'FAKE' or 'REAL'."""
    pipe = joblib.load(MODEL_PATH)
    clean = clean_text(text)
    proba = pipe.predict_proba([clean])[0]      # [P(REAL), P(FAKE)]
    fake_p = float(proba[1])
    if fake_p >= 0.5:
        return "FAKE", round(fake_p * 100, 1)
    return "REAL", round((1 - fake_p) * 100, 1)


if __name__ == "__main__":
    import sys, json
    res = train(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data", "sample.csv"))
    print(json.dumps(res, indent=2))
