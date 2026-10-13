from pathlib import Path
import joblib


# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "resolution_retriever.joblib"
)


# Load historical resolution retrieval system
bundle = joblib.load(MODEL_PATH)

vectorizer = bundle["vectorizer"]
retriever = bundle["retriever"]
tickets = bundle["tickets"]


# Prototype similarity threshold
MIN_SIMILARITY = 23.0


def find_similar_resolutions(
    ticket_text,
    max_results=3
):

    query_vector = vectorizer.transform(
        [ticket_text]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=max_results
    )

    matches = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        similarity = (
            1 - distance
        ) * 100

        if similarity < MIN_SIMILARITY:
            continue

        record = tickets.iloc[index]

        matches.append(
            {
                "similarity": similarity,
                "historical_ticket": record["text"],
                "resolution": record["answer"],
                "queue": record["queue"],
                "priority": record["priority"],
                "type": record["type"],
            }
        )

    return matches


if __name__ == "__main__":

    test_ticket = (
        "I cannot connect my laptop to the company "
        "Wi-Fi and I need internet access for work."
    )

    results = find_similar_resolutions(
        test_ticket
    )

    print("\nRESOLUTION RECOMMENDER TEST")
    print("=" * 65)

    if not results:

        print(
            "No sufficiently similar historical "
            "resolution found."
        )

    else:

        for number, match in enumerate(
            results,
            start=1
        ):

            print(
                f"\nMatch {number}: "
                f"{match['similarity']:.2f}%"
            )

            print(
                f"Queue: {match['queue']}"
            )

            print(
                f"Priority: {match['priority']}"
            )

            print("\nResolution:")

            print(
                match["resolution"]
            )