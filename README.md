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

---
