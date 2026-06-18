# EmailSort: Smart Email Classifier for Enterprises 

## 📌 Project Overview
This project builds a machine learning pipeline to classify emails into categories and predict their urgency level. The system uses text features from email subject lines and bodies, converts them into numerical representations using TF‑IDF, and trains classifiers to predict both category and urgency.

## 📂 Dataset Description
The dataset contains the following fields:

- **email_subject** → short subject line of the email  
- **email_body** → full text content of the email  
- **email_category** → target label for classification  
  - Possible values: `spam`, `inquiry`, `complaint`, `feedback`  
- **urgency level** → target label for urgency  
  - Possible values: `low`, `medium`, `high`

---

## 📝 Data Preparation
- Input features: `email_subject` and `email_body`  
- Output labels:  
  - `y1` → email category (`spam`, `inquiry`, `complaint`, `feedback`)  
  - `y2` → urgency level (`low`, `medium`, `high`)  
- The subject and body are combined into a single text string per row to capture context from both.

---

## 🔠 Text Vectorization
The combined text is transformed into numerical features using **TF‑IDF (Term Frequency–Inverse Document Frequency)**.  
This highlights important words in each email by balancing term frequency with rarity across the dataset.

Key aspects of the vectorization process:
- Limit vocabulary size for efficiency.  
- Ignore extremely rare words.  
- Ignore overly common words.  
- Include both single words (unigrams) and pairs of words (bigrams).

## **Email Category Classification Part**

## Splitting Data into Training and Testing Set for the Model Training

  Several models were trained to classify emails into categories 
  The pipeline explored both traditional machine learning algorithms and modern transformer‑based architectures:
  - **Logistic Regression** → a linear baseline classifier using TF‑IDF features.
  - **Multinomial Naïve Bayes** → a probabilistic baseline model well‑suited for text classification.
  - **BERT (bert‑base‑uncased)** → a transformer model fine‑tuned for email classification.
  - **DistilBERT** → a lighter, faster version of BERT fine‑tuned for the same task

Each model was evaluated on the same dataset split to ensure fair comparison.

## Evaluation Metrics

  Performance was measured using:
  - **Accuracy** → overall percentage of correctly classified emails.
  - **Classification Report** → detailed per‑class metrics including precision, recall, and F1‑score.


## Accuracy Score for Each models

 - DistilBERT :       97.39 %    
 - BERT       :       95.65 %     
 - Logistic Regression :  90.0 %     
 - Multinomial Naive Bayes : 82.0 %      



## 🚀 Key Insights
- Transformer‑based models (BERT, DistilBERT) significantly outperform traditional ML baselines.
- DistilBERT achieves the best trade‑off between accuracy and efficiency, making it ideal for enterprise deployment.
- Logistic Regression provides a strong baseline with lower computational cost.
- Naïve Bayes, while fast, struggles with nuanced language patterns in enterprise emails.

##  **Email Urgency Level Prediction**

### 🔎 Approach
The urgency prediction pipeline combines **rule‑based heuristics** with **machine learning models** to classify emails into `low`, `medium`, or `high` urgency.

- **Rule‑Based Layer**
  - Detects explicit urgency keywords:
    - *“urgent”*, *“immediately”*, *“asap”*, *“critical”* → **High urgency**
    - *“soon”*, *“please respond”*, *“within a week”* → **Medium urgency**
  - Acts as a pre‑classifier to boost recall for obvious urgency signals.

- **Machine Learning Layer**
  - Input: Combined subject + body text vectorized with **TF‑IDF**.
  - Models trained:
    - Logistic Regression
    - Multinomial Naïve Bayes
    - BERT (bert‑base‑uncased)
    - DistilBERT

### 📊 Evaluation Metrics
Performance was measured using:
- **Confusion Matrix** → to analyze misclassifications across urgency levels.
- **Weighted F1‑Score** → accounts for class imbalance.

### 🏆 Results
| Model                  | Weighted F1‑Score |
|-------------------------|-------------------|
| DistilBERT              | **0.95** |
| BERT                    | **0.95** |
| Logistic Regression     | 0.83 |
| Multinomial Naïve Bayes | 0.79 |

### 🧩 Key Insights
- **Transformer models (BERT, DistilBERT)** achieved the highest F1‑scores (0.95), excelling at nuanced urgency detection.
- **Logistic Regression** is a strong lightweight option with decent performance (0.83).
- **Naïve Bayes** underperformed (0.79), struggling with contextual urgency cues.
- The **rule‑based layer** improved recall for the **high urgency** class by catching explicit signals.
- Confusion matrices revealed most misclassifications occurred between **medium** and **low** urgency, reflecting subtle differences in non‑critical requests.

---

Below is the **remaining content**, written to **complete the project documentation** in a clean, academic / enterprise-ready manner. You can directly paste this after your last paragraph.

---

## 🏗️ System Architecture

The EmailSort system is designed as a modular, production-ready pipeline with clear separation of concerns:

### 1. Data Layer

* Stores raw email subject and body text.
* Maintains labeled datasets for category and urgency prediction.
* Supports incremental updates as new emails are analyzed.

### 2. Model Layer

* **Text Encoder**

  * TF-IDF vectorizer for classical ML models.
  * Transformer tokenizers for BERT and DistilBERT.
* **Category Classifier**

  * Predicts one of: spam, inquiry, complaint, feedback.
* **Urgency Classifier**

  * Predicts urgency: low, medium, high.
  * Combines ML prediction with rule-based overrides.
* **Hybrid Decision Logic**

  * Ensures high-priority signals are never missed.

### 3. Backend Layer (FastAPI)

* REST endpoint `/predict` for real-time inference.
* Handles preprocessing, inference, and post-processing.
* Returns structured JSON with confidence scores and timestamps.
* Designed to run on CPU for cost-efficient deployment.

### 4. Frontend Layer (Gradio Dashboard)

* Interactive UI for email analysis.
* PowerBI-style analytics dashboard:

  * KPIs
  * Category distribution
  * Urgency distribution
  * Email volume trends
* Supports CSV and PDF report downloads.
* Maintains session-level local memory for analytics.

---

## 🔄 End-to-End Workflow

1. User enters email subject and body in the UI.
2. Frontend sends data to the FastAPI backend.
3. Backend:

   * Cleans and preprocesses text.
   * Runs category classification.
   * Runs urgency prediction (ML + rule-based).
4. Predictions and confidence scores are returned.
5. Results are stored in local session memory.
6. Analytics dashboard updates dynamically.
7. User can export results as CSV or PDF.

---

## 📊 Explainability & Interpretability

To ensure enterprise trust and transparency:

* Confidence scores are exposed for both category and urgency.
* Rule-based urgency logic is deterministic and auditable.
* Analytics dashboards provide aggregate insights rather than black-box outputs.
* Confusion matrix analysis highlights common failure patterns.

---

## ⚙️ Deployment Strategy

### Backend

* Deployed using **Render** or similar cloud platforms.
* Uses GitHub + Git LFS for large model storage.
* CPU-only inference to reduce infrastructure cost.

### Frontend

* Deployed via:

  * Hugging Face Spaces (Gradio)
  * Or cloud VM / container platform
* Communicates securely with backend via REST API.

---

## 📈 Scalability Considerations

* Stateless API design allows horizontal scaling.
* Models can be cached in memory for faster inference.
* Batch inference supported for large email volumes.
* Can be extended with:

  * Message queues (Kafka / RabbitMQ)
  * Database persistence (PostgreSQL / MongoDB)

---

## ⚠️ Limitations

* Transformer models increase inference latency compared to classical ML.
* Medium vs low urgency remains a challenging boundary due to subtle language differences.
* Rule-based urgency keywords require periodic updates.
* Current system assumes English-only emails.

---

## 🔮 Future Enhancements

* Multilingual email support.
* Active learning for continuous model improvement.
* Email intent detection beyond fixed categories.
* SLA-based urgency scoring.
* Integration with enterprise tools (ServiceNow, Jira, Outlook, Gmail).
* Explainable AI techniques (attention visualization, SHAP).

---

## 🏁 Conclusion

EmailSort demonstrates a robust, enterprise-ready approach to intelligent email classification and urgency detection. By combining traditional NLP techniques, transformer-based deep learning models, and rule-based logic, the system achieves high accuracy, reliability, and interpretability. The modular architecture, strong evaluation results, and scalable deployment strategy make it suitable for real-world enterprise adoption.

---

**Deployment Link:** https://huggingface.co/spaces/Inzeera/EmailSort

## Created By

**Inzeera Z** ,  
Infosys Springboard Intern (27th Nov 2025 to 27th Jan 2026)

