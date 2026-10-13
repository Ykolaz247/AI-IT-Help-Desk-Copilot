import joblib

# Load retrieval bundle
bundle = joblib.load(
    "models/resolution_retriever.joblib"
)

vectorizer = bundle["vectorizer"]
retriever = bundle["retriever"]
tickets = bundle["tickets"]

MIN_SIMILARITY = 23.0

test_ticket = (
    "I cannot connect my laptop to the company Wi-Fi "
    "and I need internet access for work."
)

query_vector = vectorizer.transform(
    [test_ticket]
)

distances, indices = retriever.kneighbors(
    query_vector,
    n_neighbors=3
)

print("\nSAFE HISTORICAL RESOLUTION RETRIEVAL")
print("=" * 70)

print("\nNew Ticket:")
print(test_ticket)

accepted_matches = 0

for distance, index in zip(
    distances[0],
    indices[0]
):

    similarity = (1 - distance) * 100

    if similarity < MIN_SIMILARITY:
        continue

    accepted_matches += 1

    record = tickets.iloc[index]

    print("\n" + "-" * 70)
    print(f"Accepted Match {accepted_matches}")
    print(f"Similarity: {similarity:.2f}%")

    print("\nHistorical Ticket:")
    print(record["text"])

    print("\nQueue:")
    print(record["queue"])

    print("\nPriority:")
    print(record["priority"])

    print("\nTicket Type:")
    print(record["type"])

    print("\nSuggested Historical Resolution:")
    print(record["answer"])

if accepted_matches == 0:

    print(
        "\nNo sufficiently similar historical case was found."
    )

    print(
        "Recommendation: escalate to a human technician "
        "instead of reusing an unrelated resolution."
    )