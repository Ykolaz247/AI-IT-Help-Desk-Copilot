import joblib
import numpy as np

# Load trained model
model = joblib.load("models/queue_classifier.joblib")

test_tickets = [
    "My laptop cannot connect to the company Wi-Fi and I have an important online meeting in 20 minutes.",

    "I was charged twice for my monthly subscription. Please refund the duplicate payment.",

    "The application keeps crashing whenever I try to upload a document.",

    "Our company server has been unavailable since this morning and nobody can access the service.",

    "I would like information about your enterprise package and pricing options."
]

# Conservative confidence threshold
REVIEW_THRESHOLD = 35.0

print("\nAI IT HELPDESK ROUTING TEST")
print("=" * 65)

for number, ticket in enumerate(test_tickets, start=1):

    probabilities = model.predict_proba([ticket])[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]

    predicted_queue = classes[top_indices[0]]
    confidence = probabilities[top_indices[0]] * 100

    if confidence < REVIEW_THRESHOLD:
        status = "HUMAN REVIEW REQUIRED"
    else:
        status = "AUTO-ROUTE ELIGIBLE"

    print(f"\nTicket {number}")
    print("-" * 65)
    print(ticket)

    print(f"\nRecommended Queue: {predicted_queue}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Decision: {status}")

    print("\nTop 3 Routing Suggestions:")

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {classes[index]} "
            f"({probabilities[index] * 100:.2f}%)"
        )