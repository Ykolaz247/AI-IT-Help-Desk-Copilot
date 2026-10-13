import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Load cleaned English help-desk tickets
df = pd.read_csv(
    "data/helpdesk_tickets_english_clean.csv"
)

# Keep records with usable text and answers
df = df[
    df["text"].notna() &
    df["answer"].notna()
].copy()

df = df[
    (df["text"].str.strip() != "") &
    (df["answer"].str.strip() != "")
].reset_index(drop=True)


print("Tickets available for retrieval:", len(df))


# Convert historical ticket text to TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    sublinear_tf=True
)

ticket_vectors = vectorizer.fit_transform(
    df["text"]
)


# Build nearest-neighbor search model
retriever = NearestNeighbors(
    n_neighbors=3,
    metric="cosine",
    algorithm="brute"
)

retriever.fit(ticket_vectors)


# Save everything required for retrieval
bundle = {
    "vectorizer": vectorizer,
    "retriever": retriever,
    "tickets": df[
        [
            "text",
            "answer",
            "queue",
            "priority",
            "type"
        ]
    ]
}

joblib.dump(
    bundle,
    "models/resolution_retriever.joblib"
)


print("\nResolution retrieval system created successfully.")
print("Saved to:")
print("models/resolution_retriever.joblib")