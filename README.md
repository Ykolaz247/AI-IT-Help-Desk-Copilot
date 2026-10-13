# AI IT Help Desk Copilot

An intelligent IT support ticket triage and decision-support system built with Python, Scikit-learn, TF-IDF, machine learning, historical case retrieval, and Streamlit.

## Project Overview

The AI IT Help Desk Copilot analyzes incoming support tickets and assists IT support teams with four tasks:

1. Recommend the most appropriate support queue.
2. Predict ticket priority.
3. Flag uncertain predictions for human review.
4. Retrieve relevant resolutions from similar historical support tickets.

The system is designed as a decision-support tool rather than a fully autonomous ticketing system. Low-confidence predictions are escalated to a human technician.

## Business Problem

IT help desks often receive large volumes of support requests that must be manually:

- categorized,
- prioritized,
- routed to the correct support team,
- and investigated for possible solutions.

Manual triage can increase response time and create inconsistent routing decisions.

This project demonstrates how machine learning and historical ticket retrieval can assist support professionals while keeping humans involved in uncertain cases.

## Dataset

The original multilingual support-ticket dataset contains:

- 4,000 tickets
- 17 columns
- 5 languages

For the first version of the Copilot, the project uses:

- 1,391 English-language tickets
- ticket subject and body as text input
- 8 support queues
- 3 priority levels
- historical ticket resolutions

### Support Queues Used

- Technical Support
- Product Support
- Customer Service
- IT Support
- Billing and Payments
- Returns and Exchanges
- Service Outages and Maintenance
- Sales and Pre-Sales

## Machine Learning Pipeline

### Queue Classification

Input:

`Ticket subject + ticket body`

Processing:

`TF-IDF text vectorization`

Model:

`Logistic Regression`

Evaluation:

- Accuracy: approximately 54.6%
- Macro F1: approximately 56.4%
- Weighted F1: approximately 54.7%

The queue model was compared with:

- Logistic Regression
- Linear SVM
- Complement Naive Bayes

Logistic Regression was selected based primarily on Macro F1 and its ability to provide probability estimates for confidence-based review.

### Priority Classification

Priority classes:

- High
- Medium
- Low

Model:

`TF-IDF + Logistic Regression`

Evaluation:

- Accuracy: approximately 59.5%
- Macro F1: approximately 56.1%
- Weighted F1: approximately 59.2%

The model was compared with:

- Logistic Regression
- Linear SVM
- Complement Naive Bayes

## Human-in-the-Loop Safety

The Copilot does not automatically trust every model prediction.

Current prototype thresholds:

- Queue confidence threshold: 35%
- Priority confidence threshold: 45%

If either prediction falls below its threshold, the system returns:

`HUMAN REVIEW REQUIRED`

This reduces the risk of automatically routing uncertain tickets.

## Historical Resolution Retrieval

The project also contains a retrieval system that searches previously resolved tickets.

The system uses:

- TF-IDF
- Cosine similarity
- Nearest Neighbors

A new ticket is compared against historical tickets and the closest matching resolutions are retrieved.

Prototype minimum similarity:

`23%`

Historical resolutions are displayed only as references and must be reviewed before being applied to a new case.

## System Architecture

```text
Incoming Support Ticket
        |
        v
Text Preprocessing
        |
        +----------------------+
        |                      |
        v                      v
Queue Classifier        Priority Classifier
        |                      |
        v                      v
Queue Confidence        Priority Confidence
        |                      |
        +----------+-----------+
                   |
                   v
        Human Review Decision
                   |
                   v
      Historical Ticket Retrieval
                   |
                   v
      Resolution Recommendation
                   |
                   v
           Streamlit Interface

Technology Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- Complement Naive Bayes
- Nearest Neighbors
- Cosine Similarity
- Joblib
- Streamlit
- Anaconda
Current Features
- Automatic support-queue recommendation
- Ticket-priority prediction
- Prediction confidence scores
- Top-three routing suggestions
- Human-review escalation
- Historical case retrieval
- Historical resolution recommendations
- Similarity safety threshold
- Interactive Streamlit web interface
Project Demonstration
Streamlit Copilot Interface
The application provides a simple interface where support personnel can enter a ticket and receive routing, priority, confidence, and historical-resolution recommendations.

Confidence-Based Human Review
When prediction confidence falls below the configured thresholds, the Copilot does not treat the recommendation as an automatic decision.

Top Routing Suggestions
The system presents alternative queues and their confidence scores to support technician judgment.

Historical Resolution Retrieval
Relevant previously resolved tickets can be retrieved as reference cases. Low-similarity results are clearly marked for additional verification.

Model Development Evidence
Queue Model Comparison
Logistic Regression, Linear SVM, and Complement Naive Bayes were compared before selecting the routing model.

Priority Model Comparison
The same model-selection process was applied to ticket-priority prediction.

Data Preparation
The English-language subset was checked for missing values and duplicates before modelling.

Queue Model Training

Priority Model Training

Example Workflow
A user submits:
I cannot connect my laptop to the company Wi-Fi and I need internet access for work.

The Copilot:
1. predicts the support queue,
2. estimates its confidence,
3. predicts ticket priority,
4. determines whether human review is required,
5. searches historical resolved tickets,
6. retrieves a similar Wi-Fi case,
7. presents the historical resolution as a reference.
Limitations
This is a portfolio prototype and not a production help-desk platform.
Current limitations include:
- moderate classification accuracy,
- class imbalance,
- relatively small English training dataset,
- some low-confidence predictions,
- similarity threshold requires broader validation,
- historical resolutions may contain customer-specific or company-specific wording,
- retrieved solutions require human verification.
These limitations are intentionally exposed in the application rather than hidden.
Future Improvements
Future versions could include:
- multilingual ticket classification,
- transformer-based NLP models,
- improved class balancing,
- ticket-type prediction,
- automated knowledge-base search,
- explainable AI features,
- confidence calibration,
- REST API integration,
- ServiceNow or Jira integration,
- AWS deployment,
- authentication and user management,
- technician feedback for continuous model improvement.
Project Structure
AI_IT_Helpdesk_Copilot/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── dataset-tickets-multi-lang3-4k.csv
│   └── helpdesk_tickets_english_clean.csv
│
├── models/
│   ├── queue_classifier.joblib
│   ├── priority_classifier.joblib
│   └── resolution_retriever.joblib
│
├── src/
│   ├── train_queue_model.py
│   ├── compare_queue_models.py
│   ├── test_queue_model.py
│   ├── train_priority_model.py
│   ├── compare_priority_models.py
│   ├── test_priority_model.py
│   ├── build_resolution_retriever.py
│   ├── test_resolution_retriever.py
│   ├── resolution_engine.py
│   └── copilot_engine.py
│
├── screenshots/
│   └── portfolio/
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_EVALUATION.md
│
├── notebooks/
├── requirements.txt
└── README.md

Author
AZEEZ OLAYINKA MONSURU
Computer Science | IT Support | Machine Learning | AWS | AI Solutions