import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load cleaned dataset
df = pd.read_csv("data/helpdesk_tickets_english_clean.csv")

# Keep queues with at least 30 examples
queue_counts = df["queue"].value_counts()
valid_queues = queue_counts[queue_counts >= 30].index

df = df[df["queue"].isin(valid_queues)].copy()

print("Rows used:", len(df))
print("Queues used:", df["queue"].nunique())
print("\nQueue distribution:")
print(df["queue"].value_counts())


# Features and target
X = df["text"]
y = df["queue"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    sublinear_tf=True
)
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# Train
model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL RESULTS")
print("-------------------------")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Save trained model
joblib.dump(model, "models/queue_classifier.joblib")

print("\nModel saved to:")
print("models/queue_classifier.joblib")