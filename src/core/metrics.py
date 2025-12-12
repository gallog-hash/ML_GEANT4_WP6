"""
Metrics module for evaluating VAE generation performance.

This module provides classes for computing various evaluation metrics
to assess the quality of generated/reconstructed data.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch


class PointwiseMetrics:
    """
    Computes pointwise spatial fidelity metrics for evaluating
    reconstruction and generation quality.

    This class implements standard regression metrics to measure
    how well predictions match ground truth values on a
    point-by-point basis.

    Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Square Error
        - MAPE: Mean Absolute Percentage Error
        - R²: Coefficient of Determination

    All metrics are computed per-feature and averaged across features.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the PointwiseMetrics calculator.

        Args:
            epsilon (float): Small constant to prevent division by
                zero in percentage-based metrics. Default: 1e-8
        """
        self.epsilon = epsilon

    def _validate_inputs(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert inputs to numpy arrays.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            tuple: (y_true_array, y_pred_array) as numpy arrays

        Raises:
            ValueError: If inputs have incompatible shapes or types
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true_array = y_true.detach().cpu().numpy()
        elif isinstance(y_true, pd.DataFrame):
            y_true_array = y_true.values
        elif isinstance(y_true, np.ndarray):
            y_true_array = y_true
        else:
            raise ValueError(f"Unsupported type for y_true: {type(y_true)}")

        if isinstance(y_pred, torch.Tensor):
            y_pred_array = y_pred.detach().cpu().numpy()
        elif isinstance(y_pred, pd.DataFrame):
            y_pred_array = y_pred.values
        elif isinstance(y_pred, np.ndarray):
            y_pred_array = y_pred
        else:
            raise ValueError(f"Unsupported type for y_pred: {type(y_pred)}")

        # Validate shapes
        if y_true_array.shape != y_pred_array.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true_array.shape} "
                f"vs y_pred {y_pred_array.shape}"
            )

        return y_true_array, y_pred_array

    def compute_mae(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        per_feature: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute Mean Absolute Error (MAE).

        MAE = mean(|y_true - y_pred|)

        Args:
            y_true: Ground truth values (N, D) or (N,)
            y_pred: Predicted values (N, D) or (N,)
            per_feature: If True, return MAE per feature.
                Otherwise, return global MAE.

        Returns:
            float or np.ndarray: MAE value(s)
        """
        y_true_array, y_pred_array = self._validate_inputs(y_true, y_pred)

        absolute_errors = np.abs(y_true_array - y_pred_array)

        if per_feature and y_true_array.ndim > 1:
            # Compute per-feature (column-wise)
            return np.mean(absolute_errors, axis=0)
        else:
            # Global MAE
            return np.mean(absolute_errors)

    def compute_rmse(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        per_feature: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute Root Mean Square Error (RMSE).

        RMSE = sqrt(mean((y_true - y_pred)^2))

        Args:
            y_true: Ground truth values (N, D) or (N,)
            y_pred: Predicted values (N, D) or (N,)
            per_feature: If True, return RMSE per feature.
                Otherwise, return global RMSE.

        Returns:
            float or np.ndarray: RMSE value(s)
        """
        y_true_array, y_pred_array = self._validate_inputs(y_true, y_pred)

        squared_errors = (y_true_array - y_pred_array) ** 2

        if per_feature and y_true_array.ndim > 1:
            # Compute per-feature (column-wise)
            return np.sqrt(np.mean(squared_errors, axis=0))
        else:
            # Global RMSE
            return np.sqrt(np.mean(squared_errors))

    def compute_mape(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        per_feature: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute Mean Absolute Percentage Error (MAPE).

        MAPE = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100

        Note: epsilon is added to prevent division by zero.

        Args:
            y_true: Ground truth values (N, D) or (N,)
            y_pred: Predicted values (N, D) or (N,)
            per_feature: If True, return MAPE per feature.
                Otherwise, return global MAPE.

        Returns:
            float or np.ndarray: MAPE value(s) in percentage
        """
        y_true_array, y_pred_array = self._validate_inputs(y_true, y_pred)

        absolute_errors = np.abs(y_true_array - y_pred_array)
        absolute_true = np.abs(y_true_array) + self.epsilon

        percentage_errors = (absolute_errors / absolute_true) * 100.0

        if per_feature and y_true_array.ndim > 1:
            # Compute per-feature (column-wise)
            return np.mean(percentage_errors, axis=0)
        else:
            # Global MAPE
            return np.mean(percentage_errors)

    def compute_r2_score(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        per_feature: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Compute R² (Coefficient of Determination).

        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = sum((y_true - y_pred)^2)
            SS_tot = sum((y_true - mean(y_true))^2)

        R² = 1.0 indicates perfect prediction
        R² = 0.0 indicates prediction equals mean baseline
        R² < 0.0 indicates worse than mean baseline

        Args:
            y_true: Ground truth values (N, D) or (N,)
            y_pred: Predicted values (N, D) or (N,)
            per_feature: If True, return R² per feature.
                Otherwise, return global R².

        Returns:
            float or np.ndarray: R² value(s)
        """
        y_true_array, y_pred_array = self._validate_inputs(y_true, y_pred)

        if per_feature and y_true_array.ndim > 1:
            # Compute per-feature (column-wise)
            ss_res = np.sum((y_true_array - y_pred_array) ** 2, axis=0)
            ss_tot = np.sum(
                (y_true_array - np.mean(y_true_array, axis=0)) ** 2,
                axis=0,
            )
            # Avoid division by zero
            r2_scores = np.where(
                ss_tot > self.epsilon,
                1.0 - (ss_res / ss_tot),
                0.0,
            )
            return r2_scores
        else:
            # Global R²
            ss_res = np.sum((y_true_array - y_pred_array) ** 2)
            ss_tot = np.sum((y_true_array - np.mean(y_true_array)) ** 2)

            if ss_tot < self.epsilon:
                return 0.0

            return 1.0 - (ss_res / ss_tot)

    def compute_all(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_pred: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        per_feature: bool = True,
        feature_names: Optional[list[str]] = None,
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """
        Compute all pointwise metrics at once.

        Args:
            y_true: Ground truth values (N, D) or (N,)
            y_pred: Predicted values (N, D) or (N,)
            per_feature: If True, return metrics per feature as
                DataFrame. Otherwise, return global metrics as dict.
            feature_names: Optional list of feature names for the
                DataFrame index. Only used if per_feature=True.

        Returns:
            pd.DataFrame or dict: All metrics organized by feature
                (if per_feature=True) or as global values
                (if per_feature=False)
        """
        mae = self.compute_mae(y_true, y_pred, per_feature=per_feature)
        rmse = self.compute_rmse(y_true, y_pred, per_feature=per_feature)
        mape = self.compute_mape(y_true, y_pred, per_feature=per_feature)
        r2 = self.compute_r2_score(y_true, y_pred, per_feature=per_feature)

        if per_feature and isinstance(mae, np.ndarray):
            # Return as DataFrame
            y_true_array, _ = self._validate_inputs(y_true, y_pred)

            if feature_names is None:
                if isinstance(y_true, pd.DataFrame):
                    feature_names = list(y_true.columns)
                else:
                    feature_names = [
                        f"Feature_{i}" for i in range(y_true_array.shape[1])
                    ]

            return pd.DataFrame(
                {
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE (%)": mape,
                    "R²": r2,
                },
                index=feature_names,
            )
        else:
            # Return as dictionary
            return {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE (%)": float(mape),
                "R²": float(r2),
            }
