"""
Math utility functions for frequency application.

These functions are used for statistical calculations,
particularly in the context of logistic regression and
probability transformations.
"""

import math

from loguru import logger


def bias_for_base_rate(base_rate: float) -> float:
    """
    Convert a base fraud probability into a log-odds bias.

    Args:
        base_rate (float): Expected fraud probability in benign conditions
                           (0 < p < 1).
                           Example: 0.02 means 2% base fraud rate.

    Returns:
        float: Log-odds bias value. This is used as the intercept in the
               logistic model.

    Example:
        >>> bias_for_base_rate(0.02)
        -3.888...
    """

    # Validate input
    if not (0.0 < base_rate < 1.0):
        logger.info("Base rate must be between 0 and 1 (exclusive).")

    # Calculate log-odds
    return math.log(base_rate / (1.0 - base_rate))


def sigmoid(input_val: float) -> float:
    """
    Standard logistic sigmoid.

    Converts log-odds into probability between 0 and 1.

    Args:
        input_val (float): Log-odds value.

    Returns:
        float:
    """

    # Compute sigmoid
    return 1.0 / (1.0 + math.exp(-input_val))
