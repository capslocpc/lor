"""
A risky test case for the freq_app module.

This script imports the run_case function from freq_app.runner
and executes it with a predefined risky case dictionary.
It prints the returned result to the console.

This file is intended for testing purposes and should be run in a
controlled environment.
Use caution when executing this script, as it may involve operations
that are not safe for production environments.
"""

from loguru import logger

from freq_app.runner import run_case


# Example test case
case = {
    "Porting": "Old",
    "DarkWeb": "Medium",
    "StateMatch": "Yes",
    "ProxyFlag": "No",
    "MAID_NightDistance": "Near",
}

# Run the case and print the result
fraud_result = run_case(case)

# log the result
logger.info(f"Returned result: {fraud_result}")

# run
# poetry run python -m tests.risky_case
