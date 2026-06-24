
# 📧 EmailSort: Smart Email Classifier for Enterprises  

## 📌 Overview  
EmailSort is a machine learning project designed to **classify enterprise emails** into categories (`spam`, `inquiry`, `complaint`, `feedback`) and predict their **urgency level** (`low`, `medium`, `high`).  
The system combines **TF‑IDF vectorization**, classical ML models, and fine‑tuned transformer architectures (BERT, DistilBERT) to deliver high accuracy and actionable insights.  

---

## 📂 Dataset  
- **email_subject** → subject line of the email  
- **email_body** → full text content  
- **email_category** → target label (`spam`, `inquiry`, `complaint`, `feedback`)  
- **urgency_level** → target label (`low`, `medium`, `high`)  

---

## 🛠️ Methodology  
1. **Data Preparation**  
   - Combined subject + body for richer context.  
   - Applied text cleaning and normalization.  

2. **Vectorization**  
   - TF‑IDF with unigrams + bigrams.  
   - Vocabulary size limited for efficiency.  

3. **Models Trained**  
   - Logistic Regression  
   - Multinomial Naïve Bayes  
   - BERT (bert‑base‑uncased)  
   - DistilBERT  

4. **Urgency Prediction**  
   - Hybrid approach: rule‑based keyword detection + ML classifiers.  

---

## 📊 Results  
| Model                  | Category Accuracy | Urgency F1‑Score |
|-------------------------|------------------|-----------------|
| DistilBERT              | **97.39%**       | **0.95**        |
| BERT                    | 95.65%           | 0.95            |
| Logistic Regression     | 90.0%            | 0.83            |
| Multinomial Naïve Bayes | 82.0%            | 0.79            |

---

## 🚀 Key Insights  
- Transformer models (BERT, DistilBERT) outperform classical baselines.  
- DistilBERT offers the best balance of **accuracy + efficiency**.  
- Rule‑based layer boosts recall for **high urgency** cases.  
- Most misclassifications occur between **medium vs low urgency**.  
---

## 🎛️ Deployment  
- Built an interactive **Gradio dashboard** for real‑time predictions.  
- Dashboard includes analytics: category distribution, urgency trends, and export options (CSV/PDF).  
- Deployed on **Hugging Face Spaces** for easy access.  

🔗 **Live Demo:** [EmailSort on Hugging Face](https://huggingface.co/spaces/Inzeera/EmailSort)  

---

## 🔮 Future Enhancements  
- Multilingual email support.  
- Active learning for continuous improvement.  
- Integration with enterprise tools (ServiceNow, Jira, Outlook, Gmail).  
- Explainable AI with attention visualization.  

---

## 👩‍💻 Author  
**Inzeera Z**  
Infosys Springboard Intern (Nov 2025 – Jan 2026)  
Infosys Springboard Virtual Internship 6.0
