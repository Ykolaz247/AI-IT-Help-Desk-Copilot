import joblib
import numpy as np

from resolution_engine import find_similar_resolutions


# Load trained models
queue_model = joblib.load(
    "models/queue_classifier.joblib"
)

priority_model = joblib.load(
    "models/priority_classifier.joblib"
)


QUEUE_THRESHOLD = 35.0
PRIORITY_THRESHOLD = 45.0


def analyze_ticket(ticket_text):

    # --------------------------------
    # QUEUE PREDICTION
    # --------------------------------

    queue_probabilities = queue_model.predict_proba(
        [ticket_text]
    )[0]

    queue_classes = queue_model.classes_

    queue_indices = np.argsort(
        queue_probabilities
    )[::-1]

    predicted_queue = queue_classes[
        queue_indices[0]
    ]

    queue_confidence = (
        queue_probabilities[
            queue_indices[0]
        ] * 100
    )

    top_3_queues = []

    for index in queue_indices[:3]:

        top_3_queues.append(
            (
                queue_classes[index],
                queue_probabilities[index] * 100
            )
        )


    # --------------------------------
    # PRIORITY PREDICTION
    # --------------------------------

    priority_probabilities = (
        priority_model.predict_proba(
            [ticket_text]
        )[0]
    )

    priority_classes = (
        priority_model.classes_
    )

    priority_indices = np.argsort(
        priority_probabilities
    )[::-1]

    predicted_priority = (
        priority_classes[
            priority_indices[0]
        ]
    )

    priority_confidence = (
        priority_probabilities[
            priority_indices[0]
        ] * 100
    )


    # --------------------------------
    # HUMAN REVIEW DECISION
    # --------------------------------

    queue_review = (
        queue_confidence
        < QUEUE_THRESHOLD
    )

    priority_review = (
        priority_confidence
        < PRIORITY_THRESHOLD
    )

    if queue_review or priority_review:

        overall_decision = (
            "HUMAN REVIEW REQUIRED"
        )

    else:

        overall_decision = (
            "AUTO-ROUTE ELIGIBLE"
        )


    # --------------------------------
    # HISTORICAL RESOLUTION RETRIEVAL
    # --------------------------------

    historical_matches = (
        find_similar_resolutions(
            ticket_text,
            max_results=3
        )
    )


    return {

        "queue": predicted_queue,

        "queue_confidence":
            queue_confidence,

        "top_3_queues":
            top_3_queues,

        "priority":
            predicted_priority,

        "priority_confidence":
            priority_confidence,

        "decision":
            overall_decision,

        "historical_matches":
            historical_matches
    }


# ------------------------------------
# COMMAND-LINE TEST
# ------------------------------------

if __name__ == "__main__":

    print("\nAI IT HELPDESK COPILOT")
    print("=" * 65)

    ticket = input(
        "\nEnter support ticket:\n> "
    )

    result = analyze_ticket(ticket)

    print("\nCOPILOT ANALYSIS")
    print("=" * 65)

    print(
        f"Recommended Queue: "
        f"{result['queue']}"
    )

    print(
        f"Queue Confidence: "
        f"{result['queue_confidence']:.2f}%"
    )

    print(
        f"\nPredicted Priority: "
        f"{result['priority'].upper()}"
    )

    print(
        f"Priority Confidence: "
        f"{result['priority_confidence']:.2f}%"
    )

    print("\nTop 3 Routing Suggestions:")

    for rank, item in enumerate(
        result["top_3_queues"],
        start=1
    ):

        queue, confidence = item

        print(
            f"{rank}. {queue} "
            f"({confidence:.2f}%)"
        )

    print("\nFinal Decision:")
    print(result["decision"])


    # Historical resolution section
    print(
        "\nHistorical Resolution Suggestions:"
    )

    if not result["historical_matches"]:

        print(
            "No sufficiently similar "
            "historical case found."
        )

    else:

        for number, match in enumerate(
            result["historical_matches"],
            start=1
        ):

            print("\n" + "-" * 65)

            print(
                f"Historical Match {number}"
            )

            print(
                f"Similarity: "
                f"{match['similarity']:.2f}%"
            )

            print(
                f"Historical Queue: "
                f"{match['queue']}"
            )

            print(
                f"Historical Priority: "
                f"{match['priority']}"
            )

            print(
                "\nSuggested Resolution:"
            )

            print(
                match["resolution"]
            )