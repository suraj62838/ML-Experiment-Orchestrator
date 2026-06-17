"""
drift.py — Computes Kolmogorov-Smirnov (KS) test for numeric columns and
Chi-Square contingency test for categorical columns to detect distribution drift.
"""

from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detects statistical feature drift between a reference dataset and new incoming data.

    Parameters
    ----------
    reference_df : pd.DataFrame
        The reference baseline dataset (e.g. training set).
    threshold_ks : float
        KS statistic threshold above which numeric features are flagged as drifted.
    threshold_pval : float
        P-value threshold below which categorical features are flagged as drifted.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        threshold_ks: float = 0.1,
        threshold_pval: float = 0.05,
    ) -> None:
        self.reference_df = reference_df
        self.threshold_ks = threshold_ks
        self.threshold_pval = threshold_pval

    def detect(self, new_df: pd.DataFrame) -> dict:
        """Compare the new dataset distribution against the reference dataset.

        Parameters
        ----------
        new_df : pd.DataFrame
            The new dataset to evaluate for distribution drift.

        Returns
        -------
        dict
            DriftReport matching api schema.
        """
        logger.info("Starting drift detection on %d new samples...", len(new_df))

        # We will only compare columns that exist in both datasets
        common_cols = [c for c in new_df.columns if c in self.reference_df.columns]

        # Ignore ID and target columns if they exist
        ignore_cols = {"id", "target"}
        compare_cols = [c for c in common_cols if c.lower() not in ignore_cols]

        features_drifted = []
        drift_scores = {}

        for col in compare_cols:
            ref_col = self.reference_df[col]
            new_col = new_df[col]

            # Determine column type based on pandas dtypes
            if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(new_col):
                # KS test for numeric
                ref_clean = ref_col.dropna()
                new_clean = new_col.dropna()

                if len(ref_clean) == 0 or len(new_clean) == 0:
                    logger.warning("Column '%s' is empty, skipping KS test.", col)
                    continue

                res = ks_2samp(ref_clean, new_clean)
                stat = float(res.statistic)
                p_val = float(res.pvalue)

                drift_scores[col] = {"method": "kolmogorov-smirnov", "statistic": stat, "p_value": p_val}

                if stat > self.threshold_ks:
                    features_drifted.append(col)
                    logger.info("Feature '%s' (numeric) DRIFTED: ks_stat=%.4f (threshold=%.4f)", col, stat, self.threshold_ks)
            else:
                # Chi-Square contingency test for categorical
                ref_clean = ref_col.dropna().astype(str)
                new_clean = new_col.dropna().astype(str)

                categories = list(set(ref_clean.unique()).union(new_clean.unique()))

                if len(categories) <= 1 or len(ref_clean) == 0 or len(new_clean) == 0:
                    logger.warning("Column '%s' has insufficient categories, skipping Chi-Square test.", col)
                    continue

                # Align category frequencies
                ref_freq = ref_clean.value_counts().reindex(categories, fill_value=0).values
                new_freq = new_clean.value_counts().reindex(categories, fill_value=0).values

                obs = np.array([ref_freq, new_freq])
                # Filter categories that have 0 total count to avoid empty columns
                total_cat_counts = np.sum(obs, axis=0)
                obs = obs[:, total_cat_counts > 0]

                if obs.shape[1] <= 1:
                    p_val = 1.0
                    stat = 0.0
                else:
                    try:
                        res = chi2_contingency(obs)
                        stat = float(res.statistic)
                        p_val = float(res.pvalue)
                    except Exception as e:
                        logger.error("Chi-Square failed for column '%s': %s", col, e)
                        stat = 0.0
                        p_val = 1.0

                drift_scores[col] = {"method": "chi-square", "statistic": stat, "p_value": p_val}

                if p_val < self.threshold_pval:
                    features_drifted.append(col)
                    logger.info("Feature '%s' (categorical) DRIFTED: p_value=%.4f (threshold=%.4f)", col, p_val, self.threshold_pval)

        total_tested = len(compare_cols)
        if total_tested == 0:
            drift_percent = 0.0
        else:
            drift_percent = len(features_drifted) / total_tested

        if drift_percent == 0.0:
            status = "ok"
        elif drift_percent <= 0.30:
            status = "warning"
        else:
            status = "alert"

        logger.info("Drift detection completed: status=%s, %d/%d features drifted (%.2f%%)",
                    status, len(features_drifted), total_tested, drift_percent * 100)

        return {
            "status": status,
            "features_drifted": features_drifted,
            "drift_scores": drift_scores,
            "timestamp": datetime.now().isoformat(),
        }
