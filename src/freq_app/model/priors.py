"""
Load prior probability distributions for evidence nodes
from JSON and build TabularCPDs for the Bayesian network.
"""

import json
from pathlib import Path
from loguru import logger
from pgmpy.factors.discrete import TabularCPD

PRIORS_FILE = Path(__file__).resolve().parent.parent / "data" / "priors.json"


def load_priors():
    """
    Load priors from a JSON file.

    Returns:
        dict: Priors with states and probabilities.
    """
    if not PRIORS_FILE.exists():
        raise FileNotFoundError(f"Priors file not found at {PRIORS_FILE}")

    with open(PRIORS_FILE, "r", encoding="utf-8") as priors_file:
        priors = json.load(priors_file)

    logger.info(f"Loaded priors from {PRIORS_FILE}")
    return priors


def build_priors():
    """
    Build TabularCPDs for all priors in the JSON file.

    Returns:
        list[TabularCPD]: List of unconditional CPDs.
    """
    priors_data = load_priors()
    cpds = []

    for node_name, spec in priors_data.items():
        cpd = TabularCPD(
            variable=node_name,
            variable_card=len(spec["states"]),
            values=[[probability] for probability in spec["values"]],
            state_names={node_name: spec["states"]},
        )
        cpds.append(cpd)

    logger.info("Built TabularCPDs for priors.")
    return cpds
