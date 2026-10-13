import joblib
import numpy as np

model = joblib.load("models/priority_classifier.joblib")

test_tickets = [
    "The production server is completely down and all employees are unable to access critical systems.",

    "My laptop is running slowly, but I can still complete my work.",

    "I cannot log in to my company account and I have an important client meeting in ten minutes.",

    "Please update my mailing address in the customer profile.",

    "Several users are reporting that the payment system is failing during checkout."
]

REVIEW_THRESHOLD = 45.0

print("\nAI IT HELPDESK PRIORITY TEST")
print("=" * 65)

for number, ticket in enumerate(test_tickets, start=1):

    probabilities = model.predict_proba([ticket])[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[::-1]

    predicted_priority = classes[top_indices[0]]
    confidence = probabilities[top_indices[0]] * 100

    if confidence < REVIEW_THRESHOLD:
        status = "HUMAN REVIEW REQUIRED"
    else:
        status = "PRIORITY ACCEPTED"

    print(f"\nTicket {number}")
    print("-" * 65)
    print(ticket)

    print(f"\nPredicted Priority: {predicted_priority.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Decision: {status}")

    print("\nPriority Scores:")

    for index in top_indices:
        print(
            f"- {classes[index].upper()}: "
            f"{probabilities[index] * 100:.2f}%"
        )