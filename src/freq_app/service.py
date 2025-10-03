"""
Service layer for fraud detection application.

Handles model loading, building, and scoring.
If no model is found, it builds a new one.
"""

from pathlib import Path

from loguru import logger
from freq_app.model.model_builder import FraudBN


class FraudService:
    """Service class for fraud detection.
    Loads or builds the fraud detection model and provides scoring
    functionality.

    Attributes:
        bn (FraudBN): The Bayesian Network model for fraud detection.

    Methods:
        __init__(): Initializes the service, loading or building the model.
        score(case: dict) -> dict: Scores a single case and returns
        fraud / legit probabilities.
    """

    def __init__(self):
        """
        Initialize the FraudService.

        Loads an existing model or builds a new one if none is found.

        Raises:
            FileNotFoundError: If the model file does not exist and
            cannot be built.
        """

        # Define path for prod_model
        model_path = (
            Path(__file__).resolve().parent / "prod_model" / "prod_model.pkl"
        )

        # Ensure the directory exists
        if not model_path.exists():
            logger.info(f"Model not found: {model_path} - building new model.")

            # Build and save the model
            self.bn = FraudBN().assemble()
            self.bn.save()
        else:
            logger.info(f"Model found: {model_path} - loading the model.")

            # Load the existing model
            self.bn = FraudBN.load()

    def score(self, case: dict) -> dict:
        """
        Score a single case and return fraud/legit probabilities.
        Args:
            case (dict): A dictionary representing the case to be scored.

        Returns:
            dict: A dictionary with fraud and legit probabilities.
        """

        # Score the case using the Bayesian Network model
        return self.bn.score_case(case)
