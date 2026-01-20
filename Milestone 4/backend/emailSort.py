from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import re
import jwt
import torch
import torch.nn.functional as F

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from dotenv import load_dotenv
load_dotenv()

# ===============================
# App Configuration
# ===============================

app = FastAPI(
    title="EmailSort Backend API",
    description="Enterprise Email Classification & Urgency Detection",
    version="1.0"
)

# ===============================
# Device (IMPORTANT FOR RENDER)
# ===============================

DEVICE = torch.device("cpu")

# ===============================
# JWT Configuration
# ===============================

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ===============================
# Demo User Store
# ===============================

USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "admin123"
    }
}

# ===============================
# Model Paths
# ===============================

CATEGORY_MODEL_DIR = "models/category_model"
URGENCY_MODEL_DIR = "models/urgency_level_model"

for path in [CATEGORY_MODEL_DIR, URGENCY_MODEL_DIR]:
    if not os.path.exists(path):
        raise RuntimeError(f"Model folder missing: {path}")

# ===============================
# Labels
# ===============================

CATEGORY_LABELS = ["Complaint", "Feedback", "Spam", "Inquiry"]
URGENCY_LABELS = ["Low", "Medium", "High"]

# ===============================
# Load Tokenizer (OFFICIAL)
# ===============================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# ===============================
# Load Models (CPU SAFE)
# ===============================

category_model = DistilBertForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_DIR,
)

urgency_model = DistilBertForSequenceClassification.from_pretrained(
    URGENCY_MODEL_DIR,
)

category_model.eval()
urgency_model.eval()

# ===============================
# Schemas
# ===============================

class EmailRequest(BaseModel):
    subject: str
    body: str

class PredictionResponse(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_confidence: float
    urgency_source: str
    timestamp: str

class Token(BaseModel):
    access_token: str
    token_type: str

# ===============================
# Auth Helpers
# ===============================

def authenticate_user(username: str, password: str):
    user = USERS_DB.get(username)
    if not user or user["password"] != password:
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

# ===============================
# Text Preprocessing
# ===============================

def preprocess_text(subject: str, body: str):
    text = f"{subject} {body}".lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# Rule-Based Urgency
# ===============================

HIGH_URGENCY = [
    "urgent", "asap", "immediately", "system down",
    "not working", "failed", "critical"
]

MEDIUM_URGENCY = [
    "delay", "issue", "problem", "request", "help"
]

def rule_based_urgency(text: str) -> str:
    if any(k in text for k in HIGH_URGENCY):
        return "High"
    if any(k in text for k in MEDIUM_URGENCY):
        return "Medium"
    return "Low"

def hybrid_urgency(ml: str, rule: str) -> str:
    priority = {"Low": 0, "Medium": 1, "High": 2}
    return rule if priority[rule] > priority[ml] else ml

# ===============================
# Auth Endpoint
# ===============================

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": token, "token_type": "bearer"}

# ===============================
# Prediction Endpoint
# ===============================

@app.post("/predict", response_model=PredictionResponse)
def predict_email(
    email: EmailRequest,
    user: str = Depends(get_current_user)
):
    text = preprocess_text(email.subject, email.body)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        cat_outputs = category_model(**inputs)
        urg_outputs = urgency_model(**inputs)

    cat_probs = F.softmax(cat_outputs.logits, dim=1)
    urg_probs = F.softmax(urg_outputs.logits, dim=1)

    cat_idx = torch.argmax(cat_probs).item()
    urg_idx = torch.argmax(urg_probs).item()

    ml_urgency = URGENCY_LABELS[urg_idx]
    rule_urgency = rule_based_urgency(text)
    final_urgency = hybrid_urgency(ml_urgency, rule_urgency)

    return PredictionResponse(
        category=CATEGORY_LABELS[cat_idx],
        category_confidence=round(cat_probs[0][cat_idx].item(), 4),
        urgency=final_urgency,
        urgency_confidence=round(urg_probs[0][urg_idx].item(), 4),
        urgency_source="hybrid",
        timestamp=datetime.utcnow().isoformat()
    )

# ===============================
# Health Check
# ===============================

@app.get("/")
def health_check():
    return {"status": "EmailSort API running"}
