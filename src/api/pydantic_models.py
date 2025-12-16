from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """
    Request body schema for the prediction endpoint.
    """
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional identifier for the customer."
    )
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Dictionary of feature_name -> value. "
            "Keys must match the feature names used to train the model."
        ),
    )


class PredictionOutput(BaseModel):
    """
    Response schema for the prediction endpoint.
    """
    customer_id: Optional[str]
    is_high_risk: int
    risk_probability: float
