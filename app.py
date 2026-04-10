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
    k = int(data.get("k", 5))
    if "user_id" in data:
        result = ms.get_recommendations(requester_id=str(data["user_id"]), k=k)
    elif "skills" in data:
        result = ms.get_recommendations(
            skills=data.get("skills", {}),
            needs=data.get("needs", []),
            k=k,
        )
    else:
        return jsonify({"error": "Provide user_id OR skills+needs"}), 400
    return jsonify(result)


@app.route("/api/users", methods=["GET"])
def list_users():
    users = ms.get_all_users()
    return jsonify({"total": len(users), "users": users})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
