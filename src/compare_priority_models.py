import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import accuracy_score, f1_score


# Load cleaned English dataset
df = pd.read_csv(
    "data/helpdesk_tickets_english_clean.csv"
)

X = df["text"]
y = df["priority"]


# Same train/test split for all models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Complement Naive Bayes": ComplementNB()
}


results = []


for name, classifier in models.items():

    pipeline = Pipeline([
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

        ("classifier", classifier)
    ])


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )


    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)


print("\nPRIORITY MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)


print("\nBest model by Macro F1:")

print(
    results_df.iloc[0]["Model"]
)