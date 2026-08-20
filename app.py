"""
app.py
──────
Flask routes ONLY.
Zero ML logic here — all inference is delegated to model_service.py

Endpoints:
  GET  /                    → serves index.html
  GET  /api/health          → server status
  GET  /api/skills          → list of valid skill names
  POST /api/register        → register / update a user profile
  POST /api/recommend       → get AI-matched teammates
  GET  /api/users           → list all users (dev/debug only)
"""

import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import model_service as ms

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "running",
        "model":   "GradientBoostingRegressor",
        "skills":  ms.SKILL_NAMES,
        "db":      ms.DB_PATH,
    })


@app.route("/api/skills", methods=["GET"])
def get_skills():
    return jsonify({"skills": ms.SKILL_NAMES})


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    if not data.get("user_id"):
        return jsonify({"error": "user_id is required"}), 400
    result = ms.register_user(
        user_id = str(data["user_id"]),
        name    = data.get("name", "Anonymous"),
        major   = data.get("major", ""),
        year    = str(data.get("year", "")),
        skills  = data.get("skills", {}),
        needs   = data.get("needs", []),
    )
    return jsonify(result)


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    try:
        k = max(1, min(50, int(data.get("k", 5))))
    except (TypeError, ValueError):
        return jsonify({"error": "k must be an integer"}), 400

    candidates = data.get("candidates")
    if candidates is not None and not isinstance(candidates, list):
        return jsonify({"error": "candidates must be a list"}), 400

    if "skills" in data:
        result = ms.get_recommendations(
            skills=data.get("skills", {}),
            needs=data.get("needs", []),
            candidates=candidates,
            k=k,
        )
    elif "user_id" in data:
        result = ms.get_recommendations(requester_id=str(data["user_id"]), k=k)
    else:
        return jsonify({"error": "Provide user_id OR skills+needs"}), 400
    return jsonify(result)


@app.route("/api/users", methods=["GET"])
def list_users():
    """
    Debug helper. Off unless EXPOSE_USERS=1 is set: this returns every
    registered person's name and major, which is not something a public
    endpoint should hand out.
    """
    if os.environ.get("EXPOSE_USERS") != "1":
        return jsonify({"error": "Not available"}), 404
    users = ms.get_all_users()
    return jsonify({"total": len(users), "users": users})


if __name__ == "__main__":
    # debug=True exposes the Werkzeug console, which is remote code execution
    # on a public host. Opt in locally with FLASK_DEBUG=1, never in production.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
