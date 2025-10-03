"""
Application settings.

Uses pydantic-settings to read from environment variables and a .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings

# Define descriptive constants to avoid magic numbers
DEFAULT_PORT: int = 8000
DEFAULT_BASE_FRAUD_RATE: float = 0.02


class Settings(BaseSettings):
    """
    Application settings.

    Reads from environment variables and a .env file.

    Attributes:
        app_name (str): Name of the application.
        env (str): Environment (e.g., dev, prod).
        port (int): Port number for the application.
        build_weights (bool): Flag to indicate if weights should be built.
        raw_probs_file (str): Path to the raw probabilities file.
        base_fraud_rate (float): Base fraud rate for the model.

    Methods:
        Config: Configuration for pydantic settings.
    """

    # general
    app_name: str = Field(default="Frequency", alias="APP_NAME")
    env: str = Field(default="dev", alias="ENV")
    port: int = Field(default=DEFAULT_PORT, alias="PORT")
    build_weights: bool = Field(
        default=False,
        alias="BUILD_WEIGHTS"
    )

    raw_probs_file: str = Field(
        default="data/raw_probs.csv",
        alias="RAW_PROBS_FILE"
    )

    # model
    base_fraud_rate: float = Field(
        default=DEFAULT_BASE_FRAUD_RATE,
        alias="BASE_FRAUD_RATE"
    )

    class Config:
        """Configuration for pydantic settings.
        Reads from a .env file and environment variables.

        Attributes:
            env_file (str): Path to the .env file.
            env_file_encoding (str): Encoding of the .env file.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a singleton instance of Settings
settings = Settings()
