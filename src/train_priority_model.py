import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Load cleaned English dataset
df = pd.read_csv(
    "data/helpdesk_tickets_english_clean.csv"
)

print("Rows available:", len(df))

print("\nPriority distribution:")
print(df["priority"].value_counts())


# Input and target
X = df["text"]
y = df["priority"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Build priority classification pipeline
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


# Train model
model.fit(X_train, y_train)


# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\nPRIORITY MODEL RESULTS")
print("=" * 60)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# Save model
joblib.dump(
    model,
    "models/priority_classifier.joblib"
)

print("\nPriority model saved to:")
print(
    "models/priority_classifier.joblib"
)