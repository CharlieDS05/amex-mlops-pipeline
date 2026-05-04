from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=64)
    features: Dict[str, float] = Field(
        ..., description="Aggregated customer features"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "features": {
                    "P_2_mean": 0.45,
                    "B_1_max": 0.82,
                    "S_3_last": 0.31,
                }
            }
        }


class PredictionResponse(BaseModel):
    customer_id: str
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str
    model_version: str
    prediction_timestamp: datetime