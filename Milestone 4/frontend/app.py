import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

# ======================================================
# Backend URLs
# ======================================================
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
PREDICT_URL = f"{BACKEND_BASE_URL}/predict"
TOKEN_URL = f"{BACKEND_BASE_URL}/token"

# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="EmailSort Dashboard",
    layout="wide",
)

# ======================================================
# THEME – BLUE & WHITE
# ======================================================
st.markdown("""
            
<style>
            
html, body, [class*="css"] {
    font-size: 12px;
}
body {
    background-color: #f5f9ff;
    color: #0a2540;
}
            

section[data-testid="stSidebar"] {
    background-color: black;
    border-right: 1px solid #e5edff;
}

.nav-item {
    padding: 14px 16px;
    margin-bottom: 10px;
    border-radius: 10px;
    border: 1px solid #d6e4ff;
    font-weight: 40;
    cursor: pointer;
}

.nav-item:hover {
    background-color: #eaf1ff;
}

.nav-active {
    background-color: black
    color: white;
    border: 1px solid #0a58ff;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# MOCK DATA
# ======================================================
def load_mock_data():
    now = datetime.utcnow()
    return [
        {"Timestamp": now, "Category": "Complaint", "Category Confidence": 0.92, "Urgency": "High", "Urgency Confidence": 0.88},
        {"Timestamp": now, "Category": "Inquiry", "Category Confidence": 0.86, "Urgency": "Medium", "Urgency Confidence": 0.75},
        {"Timestamp": now, "Category": "Feedback", "Category Confidence": 0.89, "Urgency": "Low", "Urgency Confidence": 0.70},
        {"Timestamp": now, "Category": "Spam", "Category Confidence": 0.99, "Urgency": "Low", "Urgency Confidence": 0.96},
    ]

# ======================================================
# SESSION STATE
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "token" not in st.session_state:
    st.session_state.token = None

if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = load_mock_data()

# ======================================================
# LOGIN (JWT)
# ======================================================
if st.session_state.token is None:
    st.title("EmailSort Login")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        res = requests.post(
            TOKEN_URL,
            data={"username": username, "password": password}
        )
        if res.status_code == 200:
            st.session_state.token = res.json()["access_token"]
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ======================================================
# SIDEBAR NAVIGATION (CUSTOM CLICKABLE ITEMS)
# ======================================================
st.sidebar.title("Menu")
st.title("EmailSort")
st.subheader("Your smart AI-powered Email Category Classifier and Urgency Level Predictor")

def nav_item(label):
    active = "nav-active" if st.session_state.page == label else ""
    if st.sidebar.markdown(
        f'<div class="nav-item {active}">{label}</div>',
        unsafe_allow_html=True
    ):
        st.session_state.page = label

for item in ["Overview", "Email Classification", "Urgency Level", "Analytics"]:
    if st.sidebar.button(item, use_container_width=True):
        st.session_state.page = item

page = st.session_state.page

# ======================================================
# OVERVIEW (MOCK)
# ======================================================
if page == "Overview":
    st.title("Overview")

    df = pd.DataFrame(st.session_state.analytics_data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Emails Processed", len(df))
    c2.metric("High Urgency", (df["Urgency"] == "High").sum())
    c3.metric("Spam Detected", (df["Category"] == "Spam").sum())

    st.subheader("Category Distribution")
    st.bar_chart(df["Category"].value_counts())

    st.subheader("Urgency Distribution")
    st.bar_chart(df["Urgency"].value_counts())

# ======================================================
# EMAIL CLASSIFICATION (REAL BACKEND)
# ======================================================
elif page == "Email Classification":
    st.title("Category Classify")

    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body", height=220)

    if st.button("Classify Email"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res = requests.post(
            PREDICT_URL,
            json={"subject": subject, "body": body},
            headers=headers
        )

        if res.status_code == 200:
            r = res.json()
            st.success(
                f"Category: {r['category']} "
                f"({r['category_confidence']*100:.2f}%)"
            )
        else:
            st.error("Backend error")

# ======================================================
# URGENCY LEVEL (REAL BACKEND)
# ======================================================
elif page == "Urgency Level":
    st.title("Urgency Detection")

    subject = st.text_input("Email Subject", key="u1")
    body = st.text_area("Email Body", height=220, key="u2")

    if st.button("Detect Urgency"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res = requests.post(
            PREDICT_URL,
            json={"subject": subject, "body": body},
            headers=headers
        )

        if res.status_code == 200:
            r = res.json()
            st.success(
                f"Urgency: {r['urgency']} "
                f"({r['urgency_confidence']*100:.2f}%)"
            )
        else:
            st.error("Backend error")

# ======================================================
# ANALYTICS (MOCK)
# ======================================================
elif page == "Analytics":
    st.title("Analytics")

    df = pd.DataFrame(st.session_state.analytics_data)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    st.subheader("Category Distribution")
    st.bar_chart(df["Category"].value_counts())

    st.subheader("Urgency Distribution")
    st.bar_chart(df["Urgency"].value_counts())

    st.subheader("Emails Over Time")
    time_df = df.set_index("Timestamp").resample("1T").count()
    st.line_chart(time_df["Category"])

    st.subheader("Analytics Table")
    st.dataframe(df, use_container_width=True)
