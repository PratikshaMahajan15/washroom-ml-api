from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import time

# ===== FASTAPI =====
app = FastAPI()

# ===== LOAD MODEL =====
model = joblib.load("washroom_model.pkl")

FEATURES = [
    "temp_mean_12h","hum_mean_12h","gas_min_12h",
    "mq_mean_12h","motion_sum_12h","odor_mean_12h"
]

# ===== FIREBASE INIT =====
cred = credentials.Certificate("hygienemonitor-1f491-firebase-adminsdk-fbsvc-940813ae8c.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://hygienemonitor-1f491-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# ===== FEATURE EXTRACTION =====
def extract_features(data):
    df = pd.DataFrame(data.values())

    # Convert timestamp if string
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch")

    # 12h window (last 12h = 43200 sec)
    if "epoch" in df.columns:
        last_time = df["epoch"].max()
        df = df[df["epoch"] >= last_time - 43200]

    features = {
        "temp_mean_12h": df["temperature"].mean(),
        "hum_mean_12h": df["humidity"].mean(),
        "gas_min_12h": df["bmeGas"].min(),
        "mq_mean_12h": df["mq135"].mean(),
        "motion_sum_12h": df["pir"].sum(),
        "odor_mean_12h": df["bmeGas"].mean()
    }

    return features

# ===== ML INFERENCE =====
def run_inference():
    data = db.reference("/sensorLogs").get()
    if not data:
        return

    features = extract_features(data)
    X = pd.DataFrame([features])[FEATURES]

    pred = model.predict(X)[0]

    score = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        p_clean = proba[list(model.classes_).index("clean")] if "clean" in model.classes_ else 0
        p_critical = proba[list(model.classes_).index("critical")] if "critical" in model.classes_ else 0
        score = int(np.clip((p_clean*100) - (p_critical*50), 0, 100))

    insights = {
        "odorStatus": str(pred),
        "hygieneScore": score,
        "cleaningRequired": pred == "critical"
    }

    db.reference("/insights").set(insights)
    return insights

# ===== API ENDPOINTS =====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/run")
def run():
    return run_inference()

# ===== AUTO LOOP (background inference) =====
@app.on_event("startup")
def startup_event():
    def loop():
        while True:
            run_inference()
            time.sleep(15)

    import threading
    threading.Thread(target=loop, daemon=True).start()