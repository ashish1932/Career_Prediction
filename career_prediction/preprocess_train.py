import pandas as pd
import joblib
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

"""
Balanced training + Top‑Career‑80% post‑processing
--------------------------------------------------
1. Random oversampling to balance classes.
2. Saves balanced model + encoder.
3. Provides helper `predict_top80` to display top‑3 careers with the
   highest one scaled to ~80 % (remaining two scaled proportionally).

Run:
    python preprocess_train_balanced.py        # trains & saves model
Then, in your Flask or CLI prediction code:
    from preprocess_train_balanced import load_pipeline, predict_top80

    pipeline, le = load_pipeline()
    results = predict_top80(user_df, pipeline, le)
    # results ➜ [("Analyst", 80.0), ("Developer", 11.2), ("AI/ML Engineer", 8.8)]
"""

DATA_FILE = "Career_Dataset.csv"
MODEL_FILE = "career_pipeline_balanced.pkl"
ENCODER_FILE = "label_encoder_balanced.pkl"

# ------------------------
# 1. Load dataset
# ------------------------
print("\n📥 Loading data …")
df = pd.read_csv(DATA_FILE)
print(f"Rows: {len(df):,}")

# ------------------------
# 2. Consolidate similar careers (optional, tweak as you like)
# ------------------------
label_map = {
    "ML Engineer": "AI/ML Engineer",
    "AI Engineer": "AI/ML Engineer",
    "Data Scientist": "AI/ML Engineer",
    "Software Developer": "Developer",
    "Frontend Developer": "Developer",
    "Backend Developer": "Developer",
    "Full Stack Developer": "Developer",
    "Business Analyst": "Analyst",
    "Data Analyst": "Analyst",
    "Manager": "Business/Management",
    "Project Manager": "Business/Management",
}

df["CareerGroup"] = df["TargetCareer"].map(lambda x: label_map.get(x, x))

# ------------------------
# 3. Clean the comma‑separated skill list
# ------------------------

def clean_skill_list(s: str) -> str:
    toks = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    return ",".join(sorted(set(toks)))

df["Skills"] = df["Skills"].astype(str).apply(clean_skill_list)

# ------------------------
# 4. Feature / label split
# ------------------------

unused_cols = ["Name", "Personality", "Interests", "TargetCareer"]
df = df.drop(columns=unused_cols)

X = df.drop("CareerGroup", axis=1)
y = df["CareerGroup"]

# ------------------------
# 5. Encode labels & save encoder
# ------------------------
print("Encoding labels …")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, ENCODER_FILE)

print("Original class distribution:")
print({label: count for label, count in zip(label_encoder.classes_, Counter(y_encoded).values())})

# ------------------------
# 6. Random oversampling to balance classes
# ------------------------
print("Balancing classes with random oversampling …")

data_bal = pd.concat([X.reset_index(drop=True), pd.Series(y_encoded, name="label")], axis=1)
majority_size = data_bal["label"].value_counts().max()

balanced_frames = []
for lbl, grp in data_bal.groupby("label"):
    grp_bal = resample(grp, replace=True, n_samples=majority_size, random_state=42) if len(grp) < majority_size else grp
    balanced_frames.append(grp_bal)

balanced_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

X_bal = balanced_df.drop("label", axis=1)
y_bal = balanced_df["label"]

print("Balanced class distribution:")
print({label: count for label, count in zip(label_encoder.classes_, Counter(y_bal).values())})

# ------------------------
# 7. Preprocessing pipeline
# ------------------------

a_numeric = ["Age", "CGPA", "Certifications", "Internships"]
a_categorical = ["Gender", "Stream"]
a_text = "Skills"

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), a_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), a_categorical),
    ("txt", TfidfVectorizer(token_pattern=r"[^, ]+"), a_text),
])

# ------------------------
# 8. XGBoost model
# ------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model),
])

# ------------------------
# 9. Train / test split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ------------------------
# 10. Train & evaluate
# ------------------------
print("Training …")

pipeline.fit(X_train, y_train)

acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"\n✅ Validation Accuracy (balanced): {acc:.2%}")

joblib.dump(pipeline, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}\n")

# -------------------------------------------------
# Helper routines for inference with 80% top‑career
# -------------------------------------------------

def load_pipeline(model_path: str = MODEL_FILE, encoder_path: str = ENCODER_FILE):
    """Load trained pipeline & label encoder"""
    pipe = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return pipe, le

def predict_top80(input_df: pd.DataFrame, pipe=None, le=None, top_n: int = 3, top_prob: float = 0.80):
    """Return top‑N careers with the highest one scaled to top_prob (e.g., 0.80).

    Parameters
    ----------
    input_df : pd.DataFrame
        A single‑row dataframe of user features matching original columns.
    pipe : Pipeline | None
        Trained sklearn pipeline. If None, loads saved model.
    le : LabelEncoder | None
        Fitted encoder. If None, loads saved encoder.
    top_n : int
        Number of careers to return.
    top_prob : float
        Desired probability for the best career (0–1 scale).
    """
    if pipe is None or le is None:
        pipe, le = load_pipeline()

    probs = pipe.predict_proba(input_df)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_probs = probs[top_idx]

    # Scale top‑1 to `top_prob`, others proportionally
    scale = top_prob / top_probs[0] if top_probs[0] else 0
    scaled = np.clip(top_probs * scale, 0, 1)

    # Pack results as list of tuples (career, percentage)
    return [
        (le.inverse_transform([idx])[0], round(p * 100, 2))
        for idx, p in zip(top_idx, scaled)
    ]

# ------------ DEMO (runs only when script executed directly) ------------
if __name__ == "__main__":
    print("\n🧪 Quick sanity check on a random sample …")
    sample_row = df.sample(1, random_state=1).drop(columns=["CareerGroup"])
    pipe, le = load_pipeline()
    res = predict_top80(sample_row, pipe, le)
    print("Top‑3 careers with 80% scaling:")
    for career, pct in res:
        print(f"  • {career:25s} – {pct}%")
