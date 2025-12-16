from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .pydantic_models import PredictionInput, PredictionOutput

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Serve credit risk predictions from the trained model.",
    version="0.1.0",
)

# Path to the saved model artifact
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

_model: Optional[Any] = None
_feature_names: Optional[list[str]] = None


def load_model() -> None:
    """
    Load the trained model and feature names from disk.
    Called at startup.
    """
    global _model, _feature_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure you have run src/train.py and the file exists."
        )

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError(
            "Expected a dict with keys 'model' and 'feature_names' "
            f"inside {MODEL_PATH}"
        )

    if "model" not in artifact or "feature_names" not in artifact:
        raise ValueError(
            "Model artifact must contain 'model' and 'feature_names' keys."
        )

    _model = artifact["model"]
    _feature_names = list(artifact["feature_names"])


@app.on_event("startup")
def startup_event() -> None:
    """
    Load model at application startup.
    """
    try:
        load_model()
        print("Model loaded successfully for inference.")
    except Exception as exc:
        # We do not crash the app here, but mark model as unavailable.
        # The /health endpoint will show the error.
        print(f"Failed to load model: {exc}")
        # Keep _model as None so we can detect failure later.


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint.
    """
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_path": str(MODEL_PATH),
        "num_features": len(_feature_names) if _feature_names is not None else 0,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput) -> PredictionOutput:
    """
    Predict whether a customer is high risk given their features.
    """
    if _model is None or _feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded on the server. Check /health and training step.",
        )

    # Check for missing features
    missing = [f for f in _feature_names if f not in payload.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}",
        )

    # Build a DataFrame in the correct feature order
    row = [payload.features[f] for f in _feature_names]
    X = pd.DataFrame([row], columns=_feature_names)

    try:
        # model should support predict_proba
        proba = _model.predict_proba(X)[:, 1][0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {exc}",
        ) from exc

    prob_float = float(np.clip(proba, 0.0, 1.0))
    label = int(prob_float >= 0.5)

    return PredictionOutput(
        customer_id=payload.customer_id,
        is_high_risk=label,
        risk_probability=prob_float,
    )
