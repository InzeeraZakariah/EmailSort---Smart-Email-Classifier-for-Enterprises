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

The result is a sparse matrix where each row represents an email and each column represents a word or phrase feature.

---
