"""
Runner script for the Fraud Detection Service.

Can be used both from CLI and imported into other Python files.
"""

from freq_app.config.logging import get_logger
from freq_app.service import FraudService

# Initialize logger
logger = get_logger()


def run_case(case: dict) -> dict:
    """
    Run the fraud detection service for a given case.
    Args:
        case (dict): Input case dictionary
    Returns:
        dict: Fraud and Legit probabilities
    """
    service = FraudService()
    fraud_result = service.score(case)

    logger.info(f"Scored case: {case}")
    logger.success(f"Fraud probability: {fraud_result['Fraud']:.4f}")
    logger.success(f"Legit probability: {fraud_result['Legit']:.4f}")

    return fraud_result


def main():
    """
    Main function to run the fraud detection service.

    This function can be executed from the command line.
    It uses a default risky case for demonstration purposes.

    To call this function from the command line, use:
        poetry run python -m freq_app.runner

    Returns:
        None
    """





    # Default risky case (for CLI runs)
    default_case = {
        "Porting": "Recent",
        "DarkWeb": "High",
        "StateMatch": "No",
        "ProxyFlag": "Yes",
        "MAID_NightDistance": "Distant",
    }
    run_case(default_case)


if __name__ == "__main__":
    main()
