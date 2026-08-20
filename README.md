# Taazur

Taazur helps university students form project teams around **what the team is missing**, rather than around who they already know.

## The problem

Student teams usually form from existing friendships. The group ends up with four people who share the same strengths and the same gaps, and nobody notices until the project is underway.

## How it works

1. You create a profile and rate your skills.
2. You pick the skills your team still needs.
3. Taazur ranks other students by how well you and they complete each other.
4. From there you can create a team, request to join one, and keep the conversation in the app.

Teams, join requests, notifications and chat all live inside the app, so a suggestion can become a working group without leaving it.

## The recommendation approach

Matching is built on **mutual coverage**: not just whether someone has skills you lack, but whether you also cover what they are missing. A match that only works in one direction tends not to hold.

For each possible pair, the service computes six features:

| Feature | Meaning |
|---|---|
| `overlap` | shared skill mass |
| `diversity` | `1 − cosine similarity` between the two skill vectors |
| `coverage_i_j` | share of your needs the other person covers |
| `coverage_j_i` | share of their needs you cover |
| `mutual_coverage` | mean of the two coverage terms |
| `skill_balance` | difference in total skill volume |

A `GradientBoostingRegressor` scores pairs on those features. It was trained toward a compatibility target defined for the project — the sum of the two directional coverage terms — so it is best described as a **learned ranking layer over a designed target**, not a model that discovered compatibility on its own.

### What the model does and does not do

The model's output is an unbounded regression estimate, not a probability or a confidence. In practice it tracks `coverage_i_j + coverage_j_i` closely (r ≈ 0.93) with a natural range of roughly 0–2. **It is used for ordering, never shown as a percentage.**

### What the interface shows

Two things, both counted directly from the skill vectors so they can be checked against what is on screen:

- **Need coverage** — "Covers 2 of 2 skills you need", and the ring percentage. The denominator is the number of skill areas *you* selected, so covering everything you asked for is always 100%, however many needs you declare and whatever the other person needs.
- **Mutual coverage** — "you cover 2 of 2 they need", or "One-way" when you cover none of theirs.

Results are ordered by need coverage first, then by the model's score among people who cover the same amount. That is where complementarity decides: between two people who both cover everything you asked for, the one who also benefits from your skills ranks higher. Ordering purely on the model put a 50%-coverage candidate above a 100% one whenever the former also needed something you had, which is defensible internally but reads as broken next to the coverage on each card.

Several UI skills map onto one underlying area (Figma and UI/UX Design are both `uiux`), so counts are in terms of skill areas, and the search form says so.

The service also returns a structured explanation per match — which skills they bring you and at what level, and which you bring them — so a recommendation is never just a number.

## Stack

- **Frontend** — React 18 (single-page, loaded from CDN), Firebase Authentication, Cloud Firestore
- **Backend** — Flask, scikit-learn, joblib
- **Model** — GradientBoostingRegressor with a MinMaxScaler over 11 skill dimensions

Firestore is the system of record for users, teams, requests and messages. The Flask service is stateless for recommendations: the client sends the candidate pool with the request, so matching does not depend on the API's local cache being warm.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. The page is served by Flask, and the frontend points at whatever origin it was loaded from, so a local run never talks to production.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service status and loaded skill dimensions |
| `GET` | `/api/skills` | The 11 skill dimensions |
| `POST` | `/api/register` | Create or update a profile in the local cache |
| `POST` | `/api/recommend` | Ranked matches with `they_cover_you`, `you_cover_them` and an explanation. Accepts `skills` + `needs` + `candidates`, or a `user_id` |
| `GET` | `/api/users` | Debug listing. Disabled unless `EXPOSE_USERS=1` |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "skills": {"programming": 3, "ai_ml": 2},
    "needs": ["uiux", "graphic"],
    "candidates": [
      {"user_id":"u1","name":"Sara","skills":{"uiux":3,"graphic":3},"needs":["programming"]}
    ]
  }'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Port to bind |
| `FLASK_DEBUG` | unset | Set to `1` for the local reloader. Never in production |
| `EXPOSE_USERS` | unset | Set to `1` to enable the debug user listing |

No secrets are required to run the service. The Firebase config in `index.html` is the public web client config, which is designed to be shipped to browsers — access is controlled by Firestore security rules, not by hiding those values.

## Notes

Built for an Artificial Intelligence Applications course.

Skill levels are on a 1–3 scale (beginner / intermediate / advanced), matching the range the scaler was fitted on.
