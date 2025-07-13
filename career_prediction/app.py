from flask import Flask, render_template, request, send_file, jsonify
import json, os, traceback, pandas as pd
from generate_report import generate_report
from preprocess_train import load_pipeline, predict_top80

"""
Flask career‑prediction service (80 %‑scaled probabilities)
-----------------------------------------------------------
• Loads balanced XGBoost pipeline & label encoder once at startup.
• `/`            → Render form (index.html)
• `/predict`     → Return top‑3 careers (HTML view)
• `/download-report` → Generate PDF via `generate_report.py`.

Template (`index.html`) must include hidden input named `top3` to transport
JSON string of predictions for the PDF step.
"""

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────
# 1️⃣  Load model & encoder at startup
# ──────────────────────────────────────────────────────────────
print("🔄 Loading balanced model …")
PIPELINE, LABEL_ENCODER = load_pipeline()  # from preprocess_train_balanced.py
print("✅ Model ready for inference")

# ──────────────────────────────────────────────────────────────
# 2️⃣  Validation helper
# ──────────────────────────────────────────────────────────────

def validate(data: dict) -> str | None:
    """Return an error‑message string if data invalid, else None."""
    if not (10 <= data["Age"] <= 60):
        return "Invalid age (10 – 60 only)"
    if not (0.0 <= data["CGPA"] <= 10.0):
        return "CGPA must be 0–10"
    if data["Certifications"] < 0 or data["Internships"] < 0:
        return "Certifications/Internships must be non‑negative"
    if not any(c.isalpha() for c in data["Skills"] or ""):
        return "Enter at least one valid skill"
    return None

# ──────────────────────────────────────────────────────────────
# 3️⃣  Routes
# ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None, form={})


@app.route("/predict", methods=["POST"])
def predict():
    """Process form, validate, run model, render results."""
    # Collect form data
    form = {
        "name":           request.form.get("name"),
        "email":          request.form.get("email"),
        "Age":            request.form.get("age", type=int),
        "Gender":         request.form.get("gender"),
        "Stream":         request.form.get("stream"),
        "CGPA":           request.form.get("cgpa", type=float),
        "Certifications": request.form.get("certifications", type=int),
        "Internships":    request.form.get("internships", type=int),
        "Skills":         request.form.get("skills"),
    }

    # Validate user input
    error = validate(form)
    if error:
        return render_template("index.html", prediction=error, form=form)

    # Build DataFrame for model
    df_input = pd.DataFrame([{k: form[k] for k in [
        "Age", "CGPA", "Certifications", "Internships", "Gender", "Stream", "Skills"
    ]}])

    # Predict top‑3 careers with 80 % scaling
    try:
        top3 = predict_top80(df_input, PIPELINE, LABEL_ENCODER, top_n=3, top_prob=0.80)
    except Exception as exc:
        traceback.print_exc()
        return render_template("index.html", prediction=f"Prediction error: {exc}", form=form), 500

    # Ensure JSON-safe and template-safe types
    top3 = [(career, round(float(conf), 2)) for career, conf in top3]

    top3_json = json.dumps(top3)

    # Keep form values for redisplay
    form_display = {
        "name":           form["name"],
        "email":          form["email"],
        "gender":         form["Gender"],
        "stream":         form["Stream"],
        "age":            form["Age"],
        "cgpa":           form["CGPA"],
        "certifications": form["Certifications"],
        "internships":    form["Internships"],
        "skills":         form["Skills"],
    }

    return render_template(
        "index.html",
        prediction=top3[0][0],  # primary suggestion
        top3=top3,
        top3_json=top3_json,
        form=form_display,
    )


@app.route("/download-report", methods=["POST"])
def download_report():
    """Generate PDF report and send to client."""
    try:
        # Gather basic user data again (simple fields)
        user_data = {
            "name":           request.form.get("name"),
            "email":          request.form.get("email"),
            "stream":         request.form.get("stream"),
            "cgpa":           request.form.get("cgpa"),
            "certifications": request.form.get("certifications"),
            "internships":    request.form.get("internships"),
            "skills":         request.form.get("skills"),
        }

        # Prediction list arrives as JSON string in hidden input
        top3_json = request.form.get("top3", "[]")
        top3 = json.loads(top3_json)
        if not top3:
            return "Unable to generate report – no prediction data.", 400

        # Ensure reports directory
        os.makedirs("reports", exist_ok=True)
        pdf_path = generate_report(user_data, top3)
        return send_file(pdf_path, as_attachment=True, download_name="career_report.pdf")

    except Exception as e:
        print("❌ PDF generation failed:")
        traceback.print_exc()
        return f"Internal Server Error – {e}", 500


# ──────────────────────────────────────────────────────────────
# 4️⃣  Run app
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)