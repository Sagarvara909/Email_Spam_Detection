# app/app.py
from flask import Flask, render_template, request, jsonify
import joblib
import os
import math
import re
import string

# -------------------------
# Paths
# -------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "..", "models", "model.pkl")
VECT_PATH  = os.path.join(APP_ROOT, "..", "models", "vectorizer.pkl")

app = Flask(__name__, template_folder="templates")

# -------------------------
# Load model + vectorizer
# -------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# -------------------------
# Helper - preprocessing (must match training)
# -------------------------
def preprocess(text):
    if not text:
        return ""
    s = str(text).lower()

    # remove HTML
    s = re.sub(r"<.*?>", " ", s)

    # keep URLs but normalize them
    s = re.sub(r"http\S+|www\S+", " url ", s)

    # keep emails but normalize them
    s = re.sub(r"\S+@\S+", " email ", s)

    # remove non-letter except . , ! ?
    s = re.sub(r"[^a-z0-9\s.,!?]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------
# Web UI route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    probability = None

    subject = ""
    email_text = ""

    if request.method == "POST":
        subject = request.form.get("subject", "")
        email_text = request.form.get("email_text", "")

        # if trained only on body → use email_text only
        full_text = (subject + " " + email_text).strip()

        if full_text:
            # clean and vectorize
            clean_text = preprocess(full_text)
            X = vectorizer.transform([clean_text])

            # raw prediction
            pred = model.predict(X)[0]

            # probability
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0][1])
            else:
                score = model.decision_function(X)[0]
                prob = 1 / (1 + math.exp(-score))

            # threshold to avoid "everything spam"
            THRESHOLD = 0.50
            label = "SPAM" if (int(pred) == 1 and prob >= THRESHOLD) else "HAM"

            probability = round(prob, 3)

            # debug print (shows in terminal)
            print("=== DEBUG ===")
            print("Raw input:", full_text[:300].encode('utf-8', errors='ignore'))
            print("Cleaned:", clean_text[:300].encode('utf-8', errors='ignore'))
            print("Pred raw:", int(pred), "Prob:", probability, "Label after threshold:", label)
            print("Prob:", probability)
            print("Label:", label)
            print("=== /DEBUG ===")

    return render_template(
        "index.html",
        label=label,
        probability=probability,
        subject=subject,
        email_text=email_text
    )

# -------------------------
# JSON API route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text or not str(text).strip():
        return jsonify({"error": "No text provided"}), 400

    clean = preprocess(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        score = model.decision_function(X)[0]
        prob = 1 / (1 + math.exp(-score))

    THRESHOLD = 0.70
    label = "spam" if (int(pred) == 1 and prob >= THRESHOLD) else "ham"

    return jsonify({"label": label, "probability": round(prob, 3)})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
