# AI IT Help Desk Copilot — Model Evaluation

## 1. Evaluation Purpose

The project uses two supervised machine-learning models:

1. Support queue classification
2. Ticket priority classification

Multiple algorithms were compared before selecting the models used by the Copilot.

The goal was not only to maximize accuracy, but also to consider performance across classes and support confidence-based human review.

---

## 2. Queue Classification

### Dataset

After preprocessing:

- English-language tickets: 1,391
- Rows used for queue modelling: 1,361
- Training rows: 1,088
- Testing rows: 273
- Support queues: 8

### Queue Distribution

- Technical Support: 468
- Product Support: 235
- Customer Service: 213
- IT Support: 167
- Billing and Payments: 120
- Returns and Exchanges: 65
- Service Outages and Maintenance: 51
- Sales and Pre-Sales: 42

The distribution is imbalanced, with Technical Support containing substantially more examples than some smaller queues.

---

## 3. Queue Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.5458 | 0.5635 | 0.5469 |
| Linear SVM | 0.5568 | 0.5540 | 0.5505 |
| Complement Naive Bayes | 0.5201 | 0.3920 | 0.4634 |

### Selected Model

**Logistic Regression**

Although Linear SVM produced slightly higher overall accuracy, Logistic Regression produced the strongest Macro F1 score.

Macro F1 was important because the queue dataset is imbalanced and the project needs reasonable performance across both large and small support categories.

Logistic Regression also provides probability estimates through `predict_proba()`, which are required for the Copilot's confidence-based human-review mechanism.

---

## 4. Queue Classification Results

Approximate final performance:

- Accuracy: **54.6%**
- Macro F1: **56.4%**
- Weighted F1: **54.7%**

Examples of class-level performance observed during testing included stronger results for categories such as Billing and Payments and weaker performance in overlapping support categories.

This indicates that ticket routing is useful as decision support but should not be treated as fully autonomous.

---

## 5. Priority Classification

### Dataset

Priority modelling used the cleaned English ticket dataset.

Priority distribution:

- High: 595
- Medium: 557
- Low: 239

Training rows:

**1,112**

Testing rows:

**279**

Priority classes:

- High
- Medium
- Low

---

## 6. Priority Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.5950 | 0.5609 | 0.5921 |
| Linear SVM | 0.5986 | 0.5491 | 0.5891 |
| Complement Naive Bayes | 0.5269 | 0.4045 | 0.4858 |

### Selected Model

**Logistic Regression**

Linear SVM again produced slightly higher raw accuracy, but Logistic Regression achieved a higher Macro F1 score.

Logistic Regression was also preferred because it provides class probabilities needed for confidence scoring and human-review decisions.

---

## 7. Priority Classification Results

Final test performance:

- Accuracy: **59.5%**
- Macro F1: **56.1%**
- Weighted F1: **59.2%**

Class-level results included approximately:

| Priority | Precision | Recall | F1 |
|---|---:|---:|---:|
| High | 0.63 | 0.70 | 0.66 |
| Low | 0.51 | 0.46 | 0.48 |
| Medium | 0.58 | 0.54 | 0.56 |

The High-priority class performed best, while the Low class was more difficult to distinguish.

---

## 8. Confidence-Based Safety Layer

Because neither classifier is sufficiently accurate for unrestricted automated decision-making, the Copilot includes confidence thresholds.

Current prototype thresholds:

- Queue confidence threshold: **35%**
- Priority confidence threshold: **45%**

If either prediction falls below its configured threshold, the system produces:

`HUMAN REVIEW REQUIRED`

This means prediction confidence is used as an operational safety mechanism rather than allowing the model to automatically act on uncertain results.

---

## 9. Historical Resolution Retrieval Evaluation

The project also uses a retrieval system based on:

- TF-IDF
- Nearest Neighbors
- Cosine similarity

A manual Wi-Fi test produced:

- Match 1: **24.75% similarity**
- Match 2: **21.94% similarity**
- Match 3: **20.37% similarity**

The first result was a relevant laptop Wi-Fi case, while the lower-scoring results involved printer Wi-Fi problems.

Based on this initial manual test, a prototype similarity threshold of:

**23%**

was introduced.

This retained the useful laptop case and excluded the weaker printer matches.

The 23% threshold is not considered a validated production threshold. It requires evaluation across a larger labelled test set before production use.

---

## 10. Key Limitations

Important limitations include:

- relatively small English-language training set,
- class imbalance,
- overlapping meanings between support queues,
- moderate classification accuracy,
- prediction probabilities are not yet formally calibrated,
- retrieval similarity threshold was established from initial manual testing,
- historical answers may contain customer-specific wording or placeholders,
- retrieved solutions may not apply safely to every new case.

Because of these limitations, the system is intentionally designed as an AI-assisted decision-support tool rather than an autonomous help-desk system.

---

## 11. Improvement Opportunities

Future model improvements could include:

- class-weight tuning,
- oversampling minority support queues,
- hyperparameter optimization,
- transformer embeddings,
- sentence-transformer retrieval,
- confidence calibration,
- multilingual classification,
- larger training datasets,
- ticket-type modelling,
- systematic retrieval relevance evaluation,
- technician feedback loops,
- model monitoring and drift detection.

---

## Conclusion

The evaluation demonstrates that the machine-learning models provide useful support for ticket triage but are not accurate enough to replace human decision-making.

The project's main design strength is therefore the combination of:

**machine-learning recommendations + confidence scores + historical evidence + human review.**