"""
Functions to train, save, and load logistic regression weights

Weights are stored in src/freq_app/data/weights.json

Format of weights.json:
{
  "bias": float,
  "weights": {
    "feature:state": float,
    ...
  }
}
These weights are used in the logistic regression model to compute
the log-odds of fraud given the features.
"""

import json
import math

from loguru import logger
from pathlib import Path

from freq_app.config.settings import settings


# Constants  weights path
WEIGHTS_FILE = Path("src/freq_app/data/weights.json")


def logit(probability: float) -> float:
    """
    Compute the logit (log-odds) of a probability p.

    Args:
        p (float): Probability between 0 and 1 (exclusive).

    Returns:
        float: Logit value.
    """

    # Validate input
    if not (0.0 < probability < 1.0):
        logger.info(f"Probability {probability} must be between 0 and 1")
    return math.log(probability / (1 - probability))


def train_and_save_weights():
    """
    Stub training function. Later: fit logistic regression on fraud data.
    For now: derive from raw_probs.json and save to weights.json.
    """

    # Load raw_probs.json
    raw_file = Path(settings.raw_probs_file)
    # Validate it exists
    if not raw_file.exists():
        logger.info(f"Missing raw_probs.json at {raw_file}")

    # Load it
    with open(raw_file, "r") as target_file:
        raw_probs = json.load(target_file)

    # Compute bias
    bias = logit(settings.base_fraud_rate)

    # Compute per-state weights
    weights = {}
    for feature, states in raw_probs.items():
        for state, probability in states.items():
            weights[f"{feature}:{state}"] = logit(probability) - bias

    # Save them to weights.json
    payload = {"bias": bias, "weights": weights}
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, "w") as target_file:
        json.dump(payload, target_file, indent=2)

    logger.info(f" Weights written to {WEIGHTS_FILE}")
    return bias, weights


def load_weights():
    """
    Load the weights from weights.json, or train if needed.

    Load weights:
      - If BUILD_WEIGHTS=true → rebuild and overwrite.
      - If BUILD_WEIGHTS=false but weights.json missing/empty → rebuild once.
      - Else just load from weights.json.


    Returns:
        Tuple of (bias, weights dict)
        (float, Dict[str, float]): Bias and weights dictionary.
    """

    # Check if we need to build weights
    if settings.build_weights:
        return train_and_save_weights()

    # Check if weights.json exists and is non-empty
    if not WEIGHTS_FILE.exists() or WEIGHTS_FILE.stat().st_size == 0:
        logger.info("No weights.json found (or empty) → building fresh weights")
        return train_and_save_weights()

    # Load existing weights
    with open(WEIGHTS_FILE, "r") as target_file:
        weights_data = json.load(target_file)

    # Return bias and weights
    return weights_data["bias"], weights_data["weights"]


# Initialize model parameters (bias + weights) at module load.
# This makes them immediately available to other modules via:
#     from freq_app.model.weights import bias, WEIGHTS
bias, WEIGHTS = load_weights()
