import os
import re
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ===============================
# Model Paths
# ===============================
CATEGORY_MODEL_DIR = "models/category_model"
URGENCY_MODEL_DIR = "models/urgency_level_model"

for path in [CATEGORY_MODEL_DIR, URGENCY_MODEL_DIR]:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing model folder: {path}")

# ===============================
# Labels
# ===============================
CATEGORY_LABELS = ["Complaint", "Feedback", "Spam", "Inquiry"]
URGENCY_LABELS = ["Low", "Medium", "High"]

# ===============================
# Load Tokenizer & Models
# ===============================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

category_model = DistilBertForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_DIR
)
urgency_model = DistilBertForSequenceClassification.from_pretrained(
    URGENCY_MODEL_DIR
)

category_model.eval()
urgency_model.eval()

# ===============================
# Text Preprocessing
# ===============================
def preprocess_text(subject, body):
    text = f"{subject} {body}".lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Rule-Based Urgency
# ===============================
HIGH_URGENCY = ["urgent", "asap", "immediately", "system down", "failed", "critical"]
MEDIUM_URGENCY = ["delay", "issue", "problem", "request", "help"]

def rule_based_urgency(text):
    if any(k in text for k in HIGH_URGENCY):
        return "High"
    if any(k in text for k in MEDIUM_URGENCY):
        return "Medium"
    return "Low"

def hybrid_urgency(ml, rule):
    priority = {"Low": 0, "Medium": 1, "High": 2}
    return rule if priority[rule] > priority[ml] else ml

# ===============================
# Email Analysis
# ===============================
def analyze_email(subject, body, history):
    if not subject or not body:
        raise gr.Error("Subject and Body are required")

    text = preprocess_text(subject, body)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        cat_logits = category_model(**inputs).logits
        urg_logits = urgency_model(**inputs).logits

    cat_probs = F.softmax(cat_logits, dim=1)
    urg_probs = F.softmax(urg_logits, dim=1)

    cat_idx = torch.argmax(cat_probs).item()
    urg_idx = torch.argmax(urg_probs).item()

    ml_urgency = URGENCY_LABELS[urg_idx]
    rule_urg = rule_based_urgency(text)
    final_urgency = hybrid_urgency(ml_urgency, rule_urg)

    record = {
        "subject": subject,
        "body": body,
        "category": CATEGORY_LABELS[cat_idx],
        "category_confidence": round(cat_probs[0][cat_idx].item(), 4),
        "urgency": final_urgency,
        "urgency_confidence": round(urg_probs[0][urg_idx].item(), 4),
        "timestamp": datetime.utcnow().isoformat()
    }

    history.append(record)

    result_md = f"""
### 📧 Email Analysis Result

**Category:** `{record['category']}`  
**Category Confidence:** `{record['category_confidence'] * 100:.2f}%`

**Urgency:** `{record['urgency']}`  
**Urgency Confidence:** `{record['urgency_confidence'] * 100:.2f}%`

**Timestamp:** `{record['timestamp']}`
"""

    return result_md, history

# ===============================
# Analytics
# ===============================
def generate_analytics(history, category_filter, urgency_filter):
    if not history:
        return "No data available", None, None, None

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if category_filter != "All":
        df = df[df["category"] == category_filter]

    if urgency_filter != "All":
        df = df[df["urgency"] == urgency_filter]

    if df.empty:
        return "No data after applying filters", None, None, None

    total = len(df)
    top_category = df["category"].mode()[0]
    high_pct = round(
        (df["urgency"].value_counts().get("High", 0) / total) * 100, 2
    )

    kpi = f"""
## 📊 Key Metrics
- **Total Emails:** {total}
- **Top Category:** {top_category}
- **High Urgency %:** {high_pct}%
"""

    fig1, ax1 = plt.subplots()
    df["category"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Category Distribution")

    fig2, ax2 = plt.subplots()
    df["urgency"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
        ax=ax2
    )
    ax2.set_title("Urgency Distribution")

    fig3, ax3 = plt.subplots()
    df.groupby(df["timestamp"].dt.date).size().plot(
        kind="line", marker="o", ax=ax3
    )
    ax3.set_title("Email Volume Over Time")

    return kpi, fig1, fig2, fig3

# ===============================
# Downloads
# ===============================
def download_csv(history):
    if not history:
        return None
    path = "email_results.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path

def download_pdf(history):
    if not history:
        return None

    path = "email_report.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    text = c.beginText(40, 800)
    text.setFont("Helvetica", 10)

    text.textLine("EmailSort – Analytics Report")
    text.textLine("-" * 70)

    for i, r in enumerate(history, 1):
        text.textLine(f"{i}. {r['subject']}")
        text.textLine(f"   Category: {r['category']} ({r['category_confidence']})")
        text.textLine(f"   Urgency: {r['urgency']} ({r['urgency_confidence']})")
        text.textLine(f"   Time: {r['timestamp']}")
        text.textLine("")

    c.drawText(text)
    c.save()
    return path

# ===============================
# Gradio UI
# ===============================
with gr.Blocks(title="EmailSort AI Dashboard") as demo:

    history_state = gr.State([])

    gr.Markdown("# 📬 EmailSort – AI Classification & Analytics")

    with gr.Tabs():

        # -------- Analyze --------
        with gr.Tab("Analyze Email"):
            subject = gr.Textbox(label="Email Subject")
            body = gr.Textbox(label="Email Body", lines=6)
            result = gr.Markdown()

            gr.Button("Analyze").click(
                analyze_email,
                inputs=[subject, body, history_state],
                outputs=[result, history_state]
            )

        # -------- Analytics --------
        with gr.Tab("Analytics"):
            category_filter = gr.Dropdown(
                ["All"] + CATEGORY_LABELS, value="All", label="Category"
            )
            urgency_filter = gr.Dropdown(
                ["All"] + URGENCY_LABELS, value="All", label="Urgency"
            )

            kpi = gr.Markdown()
            fig1 = gr.Plot()
            fig2 = gr.Plot()
            fig3 = gr.Plot()

            gr.Button("Refresh Analytics").click(
                generate_analytics,
                inputs=[history_state, category_filter, urgency_filter],
                outputs=[kpi, fig1, fig2, fig3]
            )

        # -------- Downloads --------
        with gr.Tab("Download"):
            csv_file = gr.File(label="CSV File")
            pdf_file = gr.File(label="PDF Report")

            gr.Button("Download CSV").click(
                download_csv, history_state, csv_file
            )

            gr.Button("Download PDF").click(
                download_pdf, history_state, pdf_file
            )

demo.launch()
