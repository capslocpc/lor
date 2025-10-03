"""
A module to build and manage a Bayesian Network for fraud detection.

Builds, saves, loads, and scores the Bayesian Network model for
fraud detection. Uses pgmpy for Bayesian Network operations.
"""

from __future__ import annotations

import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List

from loguru import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from freq_app.model.priors import build_priors, load_priors
from freq_app.model.weights import bias, WEIGHTS
from freq_app.utils.math_utils import sigmoid


class FraudBN:
    """
    Build, save, load, and score a Bayesian Network for fraud detection.

    Structure:
        Porting, DarkWeb, StateMatch, ProxyFlag, MAID_NightDistance -> Fraud

    Methods:
        - build_fraud_cpd(): construct Fraud CPD from logistic weights.
        - assemble(): add prior CPDs + Fraud CPD and validate the model.
        - score_case(evidence): return P(Fraud), P(Legit) given evidence.
        - save(): persist model to src/freq_app/prod_model/prod_model.pkl.
        - load(): load model from the same path.
    """

    def __init__(self) -> None:
        """Initialize the BN structure (nodes + edges)."""
        self.model = DiscreteBayesianNetwork(
            [
                ("Porting", "Fraud"),
                ("DarkWeb", "Fraud"),
                ("StateMatch", "Fraud"),
                ("ProxyFlag", "Fraud"),
                ("MAID_NightDistance", "Fraud"),
            ]
        )

    def build_fraud_cpd(self) -> TabularCPD:
        """
        Build the CPD for the Fraud node from interpretable logistic weights.

        The probability is computed as:
            p = sigmoid(bias + sum(weight[state] for each parent state))

        Returns:
            TabularCPD: CPD for variable "Fraud" with states ["Fraud", "Legit"]
        """
        priors_data = load_priors()
        porting_states = priors_data["Porting"]["states"]
        darkweb_states = priors_data["DarkWeb"]["states"]
        statematch_states = priors_data["StateMatch"]["states"]
        proxy_states = priors_data["ProxyFlag"]["states"]
        maid_states = priors_data["MAID_NightDistance"]["states"]

        p_fraud_yes: List[float] = []
        for p_state, d_state, s_state, x_state, m_state in product(
            porting_states, darkweb_states, statematch_states, proxy_states, maid_states
        ):
            logit = (
                bias +
                WEIGHTS[f"Porting:{p_state}"] +
                WEIGHTS[f"DarkWeb:{d_state}"] +
                WEIGHTS[f"StateMatch:{s_state}"] +
                WEIGHTS[f"ProxyFlag:{x_state}"] +
                WEIGHTS[f"MAID_NightDistance:{m_state}"]
            )
            p_fraud_yes.append(sigmoid(logit))

        fraud_cpd_matrix = [
            p_fraud_yes,                           # P(Fraud = "Fraud")
            [1.0 - prob for prob in p_fraud_yes],  # P(Fraud = "Legit")
        ]

        return TabularCPD(
            variable="Fraud",
            variable_card=2,
            values=fraud_cpd_matrix,
            evidence=[
                "Porting",
                "DarkWeb",
                "StateMatch",
                "ProxyFlag",
                "MAID_NightDistance",
            ],
            evidence_card=[3, 4, 2, 2, 4],
            state_names={
                "Fraud": ["Fraud", "Legit"],
                "Porting": porting_states,
                "DarkWeb": darkweb_states,
                "StateMatch": statematch_states,
                "ProxyFlag": proxy_states,
                "MAID_NightDistance": maid_states,
            },
        )

    def assemble(self) -> "FraudBN":
        """
        Add prior CPDs and Fraud CPD, then validate the model.
        Returns:
            FraudBN: self (builder) so you can chain .score_case(...)
        """
        fraud_cpd = self.build_fraud_cpd()
        prior_cpds = build_priors()
        self.model.add_cpds(*prior_cpds, fraud_cpd)
        assert self.model.check_model(), "Model/CPDs inconsistent!"
        logger.info("Bayesian Network assembled and validated.")
        return self

    def score_case(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Score a case given evidence.

        Args:
            evidence: dict like:
                {
                    "Porting": "Recent",
                    "DarkWeb": "High",
                    "StateMatch": "No",
                    "ProxyFlag": "Yes",
                    "MAID_NightDistance": "Distant"
                }

        Returns:
            dict: {"Fraud": p_fraud, "Legit": p_legit}
        """
        infer = VariableElimination(self.model)
        fraud_distribution = infer.query(
            variables=["Fraud"],
            evidence=evidence,
            show_progress=False,
        )
        fraud_score = dict(zip(
            fraud_distribution.state_names["Fraud"],
            fraud_distribution.values)
        )
        return {
            "Fraud": float(fraud_score["Fraud"]),
            "Legit": float(fraud_score["Legit"])
        }

    def save(self) -> Path:
        """
        Save the model to src/freq_app/prod_model/prod_model.pkl.
        Creates the directory if missing.
        """
        models_dir = (
            Path(__file__).resolve().parent.parent / "prod_model"
        )
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "prod_model.pkl"
        with open(model_path, "wb") as file_handle:
            pickle.dump(self.model, file_handle)

        logger.info(f"Model saved to {model_path}")
        return model_path

    @classmethod
    def load(cls) -> "FraudBN":
        """
        Load the model from src/freq_app/prod_model/prod_model.pkl.

        Returns:
            FraudBN: instance with `model` populated.
        """
        models_dir = (
            Path(__file__).resolve().parent.parent / "prod_model"
        )
        model_path = models_dir / "prod_model.pkl"

        with open(model_path, "rb") as file_handle:
            loaded_model = pickle.load(file_handle)

        fraud_bn_obj = cls()
        fraud_bn_obj.model = loaded_model
        logger.info(f"Model loaded from {model_path}")
        return fraud_bn_obj
