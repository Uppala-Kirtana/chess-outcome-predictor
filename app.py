"""
app.py — FastAPI service that serves chess outcome predictions.

Loads the trained model once at startup, then exposes a /predict
endpoint that takes game details and returns a predicted outcome.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# --- Load the model once, when the server starts ---
# (Not inside the endpoint function — loading from disk on every
# request would be slow and wasteful.)
bundle = joblib.load("model.joblib")
model = bundle["model"]
opening_encoder = bundle["opening_encoder"]
top_openings = bundle["top_openings"]
feature_cols = bundle["feature_cols"]

app = FastAPI(title="Chess Outcome Predictor")


# --- Define what a valid request looks like ---
# Pydantic validates incoming requests automatically — if someone sends
# a string where a number is expected, FastAPI rejects it before our
# code even runs, with a clear error message.
class GameRequest(BaseModel):
    white_rating: int
    black_rating: int
    opening_name: str
    base_time: int
    increment: int
    turns: int


@app.get("/")
def root():
    return {"status": "Chess outcome predictor is running"}


@app.post("/predict")
def predict(game: GameRequest):
    # Recreate the exact same feature engineering we did in train.py.
    # This step has to match training exactly, or the model gets
    # confused by differently-shaped input than it learned from.
    rating_diff = game.white_rating - game.black_rating

    # If this opening wasn't in our top 20 during training, treat it
    # as "Other" — same rule we applied when training.
    opening_grouped = game.opening_name if game.opening_name in top_openings else "Other"
    opening_encoded = opening_encoder.transform([opening_grouped])[0]

    features = pd.DataFrame(
        [[rating_diff, game.base_time, game.increment, opening_encoded, game.turns]],
        columns=feature_cols,
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Map each class label to its predicted probability, e.g. {"white": 0.62, ...}
    prob_dict = dict(zip(model.classes_, probabilities.round(3)))

    return {
        "predicted_winner": prediction,
        "probabilities": prob_dict,
    }
