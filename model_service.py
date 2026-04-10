"""
model_service.py
────────────────
Responsible for:
  - Loading the trained model and scaler (once, at startup)
  - User registration (storing skill/need vectors)
  - Computing pairwise features
  - Running inference and returning ranked matches
  - Generating explanations

NO training code. NO CSV. NO dataset logic.
"""

import sqlite3
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

os.makedirs("data", exist_ok=True)
# ── Config ───────────────────────────────────────────────────
MODEL_PATH  = "partner_match_model-2_fixed.pkl"
SCALER_PATH = "skill_scaler_fixed.pkl"
DB_PATH     = "data/taazur.db"

# The 11 skill dimensions the model was trained on (ORDER MATTERS)
SKILL_NAMES = [
    "programming", "uiux", "graphic", "content", "marketing",
    "project_mgmt", "data_analysis", "ai_ml", "research",
    "presentation", "video"
]

# ── Load model once at import time ───────────────────────────
model        = joblib.load(MODEL_PATH)
skill_scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded.")

# ── Database setup ───────────────────────────────────────────

def init_db():
    """Create users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            name        TEXT,
            major       TEXT,
            year        TEXT,
            skills_json TEXT,   -- JSON: {"programming": 0.8, ...}
            needs_json  TEXT    -- JSON: ["uiux", "ai_ml"]
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Internal helpers ─────────────────────────────────────────
def _to_vector(skills_dict: dict) -> np.ndarray:
    """Convert skill dict → raw numpy vector (11 values)."""
    return np.array(
        [float(skills_dict.get(s, 0.0)) for s in SKILL_NAMES],
        dtype=float
    )

def _scale(raw_vector: np.ndarray) -> np.ndarray:
    """Apply the same MinMaxScaler used during training."""
    return skill_scaler.transform(raw_vector.reshape(1, -1))[0]

def _needs_vector(needs_list: list) -> np.ndarray:
    """Convert list of skill names → binary vector (11 values)."""
    return np.array(
        [1.0 if s in needs_list else 0.0 for s in SKILL_NAMES],
        dtype=float
    )

def _compute_features(E_i, N_i, E_j, N_j) -> list:
    """
    Compute the 6 pairwise features the GradientBoostingRegressor
    was trained on. This is inference-only — no training logic.

    Features (in order):
      1. overlap          — shared skill mass
      2. diversity        — 1 - cosine_similarity
      3. coverage_i_j     — fraction of i's needs that j covers
      4. coverage_j_i     — fraction of j's needs that i covers
      5. mutual_coverage  — average of coverage_i_j and coverage_j_i
      6. skill_balance    — absolute difference in skill volume
    """
    def _coverage(n, e):
        total = np.sum(n)
        return float(np.sum((n * e) > 0) / total) if total > 0 else 0.0

    cov_ij = _coverage(N_i, E_j)
    cov_ji = _coverage(N_j, E_i)

    return [
        float(np.sum(E_i * E_j)),                              # overlap
        1.0 - float(cosine_similarity([E_i], [E_j])[0][0]),    # diversity
        cov_ij,                                                 # coverage_i_j
        cov_ji,                                                 # coverage_j_i
        (cov_ij + cov_ji) / 2.0,                               # mutual_coverage
        float(abs(np.sum(E_i) - np.sum(E_j))),                 # skill_balance
    ]

def _build_explanation(E_i, N_i, E_j, N_j) -> dict:
    """
    Explain why user j is a good match for user i.
    Returns a structured dict (not a string) so the frontend
    can render it however it wants.
    """
    provides_to_you = []
    for idx, skill in enumerate(SKILL_NAMES):
        if N_i[idx] == 1 and E_j[idx] > 0:
            level = ("advanced"     if E_j[idx] > 0.6 else
                     "intermediate" if E_j[idx] > 0.3 else "beginner")
            provides_to_you.append({"skill": skill, "level": level})

    you_provide = []
    for idx, skill in enumerate(SKILL_NAMES):
        if N_j[idx] == 1 and E_i[idx] > 0:
            level = ("advanced"     if E_i[idx] > 0.6 else
                     "intermediate" if E_i[idx] > 0.3 else "beginner")
            you_provide.append({"skill": skill, "level": level})

    div = 1.0 - float(cosine_similarity([E_i], [E_j])[0][0])
    return {
        "provides_to_you": provides_to_you,
        "you_provide":     you_provide,
        "high_diversity":  div > 0.5,
        "low_redundancy":  float(np.sum(E_i * E_j)) < 1.0,
    }

# ── Public API ────────────────────────────────────────────────

def register_user(user_id: str, name: str, major: str, year: str,
                  skills: dict, needs: list) -> dict:
    """
    Save or update a user in the database.

    skills: {"programming": 0.8, "uiux": 0.3, ...}
    needs:  ["ai_ml", "graphic"]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO users (user_id, name, major, year, skills_json, needs_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name=excluded.name, major=excluded.major, year=excluded.year,
            skills_json=excluded.skills_json, needs_json=excluded.needs_json
    """, (user_id, name, major, year,
          json.dumps(skills), json.dumps(needs)))
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()

    return {"status": "ok", "total_users": count}


def get_recommendations(requester_id: str = None,
                        skills: dict = None,
                        needs: list = None,
                        k: int = 5) -> dict:
    """
    Return top-k matches for a user.

    Can be called two ways:
      A) By user_id  → looks up their stored profile
      B) By skills+needs directly (guest search, no registration needed)
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT user_id, name, major, year, skills_json, needs_json FROM users"
    ).fetchall()
    conn.close()

    if len(rows) == 0:
        return {"status": "no_users", "matches": []}

    # Get requester's vectors
    if requester_id:
        me = next((r for r in rows if r[0] == requester_id), None)
        if me is None:
            return {"status": "error", "message": "user_id not found"}, 404
        my_skills  = json.loads(me[4])
        my_needs   = json.loads(me[5])
        candidates = [r for r in rows if r[0] != requester_id]
    else:
        my_skills  = skills or {}
        my_needs   = needs or []
        candidates = rows

    if len(candidates) == 0:
        return {"status": "no_candidates",
                "message": "No other users registered yet.",
                "matches": []}

    # Build requester vectors
    E_me = _scale(_to_vector(my_skills))
    N_me = _needs_vector(my_needs)

    # Score every candidate
    results = []
    for row in candidates:
        uid, name, major, year, s_json, n_json = row
        E_j = _scale(_to_vector(json.loads(s_json)))
        N_j = _needs_vector(json.loads(n_json))

        features = _compute_features(E_me, N_me, E_j, N_j)
        score    = float(model.predict([features])[0])

        results.append({
            "user_id":         uid,
            "name":            name,
            "major":           major,
            "year":            year,
            "score_raw":       score,
            "score_percent":   min(100, max(0, round(score * 25))),
            "dominant_skill":  SKILL_NAMES[int(np.argmax(E_j))],
            "skills":          json.loads(s_json),
            "needs":           json.loads(n_json),
            "mutual_coverage": round(features[4], 3),
            "diversity":       round(features[1], 3),
            "explanation":     _build_explanation(E_me, N_me, E_j, N_j),
        })

    # Sort by score descending, take top-k
    results.sort(key=lambda x: x["score_raw"], reverse=True)
    top_k = results[:k]

    # Clean up score_raw before sending to frontend
    for i, r in enumerate(top_k):
        r["rank"] = i + 1
        del r["score_raw"]

    return {
        "status":           "ok",
        "total_candidates": len(candidates),
        "matches":          top_k,
    }


def get_all_users() -> list:
    """Return lightweight list of all registered users (for debugging)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT user_id, name, major, year FROM users"
    ).fetchall()
    conn.close()
    return [{"user_id": r[0], "name": r[1], "major": r[2], "year": r[3]}
            for r in rows]
