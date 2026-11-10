import pickle
from pathlib import Path
from typing import Literal

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import build_matchup_features, matchup_warnings

# 1. Load the model once, when the service starts
MODEL_FILE = Path(__file__).parent / "model.bin"
with open(MODEL_FILE, "rb") as f_in:
    dv, model, snapshot = pickle.load(f_in)

FEATURE_NAMES = list(dv.get_feature_names_out())

DATA_UNTIL = snapshot["last_fight_date"].max().date().isoformat()

app = FastAPI(title="UFC Fight Predictor")


# 2. What the user sends
class FightRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    weight_class: str | None = None
    is_title: Literal[0, 1] | None = None
    scheduled_rounds: Literal[3, 5] | None = None


# 3. What the service returns
class FightResponse(BaseModel):
    fighter_a: str
    fighter_b: str
    fighter_a_win_prob: float
    fighter_b_win_prob: float
    predicted_winner: str
    warnings: list[str]


def predict_proba(row):
    X = dv.transform([row])
    return float(model.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES))[0])


@app.get("/health")
def health():
    return {"status": "ok", "data_until": DATA_UNTIL}

@app.get("/fighters")
def fighters():
    df = snapshot[["FIGHTER", "weight_class", "divisions"]].sort_values("FIGHTER")
    return df.to_dict(orient="records")


@app.post("/predict", response_model=FightResponse)
def predict(req: FightRequest):
    # Optional context: keep only the fields the user actually sent
    context = {
        "weight_class": req.weight_class,
        "is_title": req.is_title,
        "scheduled_rounds": req.scheduled_rounds,
    }
    context = {k: v for k, v in context.items() if v is not None}

    # Build the features from both sides (A vs B and B vs A)
    try:
        row_ab = build_matchup_features(snapshot, req.fighter_a, req.fighter_b, context)
        row_ba = build_matchup_features(snapshot, req.fighter_b, req.fighter_a, context)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=e.args[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=e.args[0])

    # Symmetric prediction: the order of the names does not matter
    p_ab = predict_proba(row_ab)          # P(A beats B)
    p_ba = predict_proba(row_ba)          # P(B beats A)
    prob_a = (p_ab + (1 - p_ba)) / 2

    return FightResponse(
        fighter_a=req.fighter_a,
        fighter_b=req.fighter_b,
        fighter_a_win_prob=round(prob_a, 3),
        fighter_b_win_prob=round(1 - prob_a, 3),
        predicted_winner=req.fighter_a if prob_a >= 0.5 else req.fighter_b,
        warnings=matchup_warnings(snapshot, req.fighter_a, req.fighter_b),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)