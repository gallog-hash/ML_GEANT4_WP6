"""
Bragg Peak-specific metrics for hadrontherapy simulation evaluation.

This module provides domain-specific metrics for evaluating the physical
accuracy of Bragg peak reconstruction in particle therapy simulations.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


class BraggPeakMetrics:
    """
    Computes Bragg Peak-specific metrics for hadrontherapy evaluation.

    This class implements domain-specific metrics to assess the physical
    accuracy of generated Bragg peak profiles:

    Metrics:
        - Peak Position Error: Difference in depth of maximum dose (mm)
        - Peak Height Error: Difference in maximum dose value (feature units)
        - Peak FWHM Error: Difference in Full Width at Half Maximum (mm)
        - Distal Falloff Error: MAE of normalized dose curves (dimensionless)

    All metrics are computed per-feature (e.g., per particle type).

    Units:
        - Spatial metrics (position, FWHM) use the spatial coordinate units
        - Height error uses the same units as the analyzed feature
        - Falloff error is dimensionless (normalized by peak height)
    """

    def __init__(
        self,
        spatial_coordinate: str = "x",
        falloff_range_percent: float = 0.8,
        min_peak_prominence: float = 0.1,
    ):
        """
        Initialize the BraggPeakMetrics calculator.

        Args:
            spatial_coordinate (str): Name of the spatial coordinate
                column (e.g., 'x', 'depth'). Default: 'x'
            falloff_range_percent (float): Percentage of peak height
                to define distal falloff region (e.g., 0.8 means from
                80% down to baseline). Default: 0.8
            min_peak_prominence (float): Minimum prominence for peak
                detection as fraction of max value. Default: 0.1
        """
        self.spatial_coordinate = spatial_coordinate
        self.falloff_range_percent = falloff_range_percent
        self.min_peak_prominence = min_peak_prominence

    def _validate_inputs(
        self,
        spatial_true: np.ndarray,
        y_true: np.ndarray,
        spatial_pred: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Validate input arrays.

        Args:
            spatial_true: Spatial coordinates for ground truth
            y_true: Ground truth values
            spatial_pred: Spatial coordinates for predictions
            y_pred: Predicted values

        Raises:
            ValueError: If inputs have invalid shapes or values
        """
        if len(spatial_true) != len(y_true):
            raise ValueError(
                f"spatial_true length ({len(spatial_true)}) != "
                f"y_true length ({len(y_true)})"
            )
        if len(spatial_pred) != len(y_pred):
            raise ValueError(
                f"spatial_pred length ({len(spatial_pred)}) != "
                f"y_pred length ({len(y_pred)})"
            )
        if len(spatial_true) < 3 or len(spatial_pred) < 3:
            raise ValueError("Need at least 3 points for peak detection")

    def _find_peak_position(
        self, spatial: np.ndarray, values: np.ndarray
    ) -> Tuple[float, float, int]:
        """
        Find the primary peak position and height.

        Args:
            spatial: Spatial coordinates (e.g., depth)
            values: Dose/LET values

        Returns:
            tuple: (peak_position, peak_height, peak_index)
        """
        # Find peaks with minimum prominence
        prominence_threshold = self.min_peak_prominence * np.max(values)
        peaks, properties = find_peaks(values, prominence=prominence_threshold)

        if len(peaks) == 0:
            # No peaks found, use global maximum
            peak_idx = np.argmax(values)
        else:
            # Use the most prominent peak
            prominences = properties["prominences"]
            peak_idx = peaks[np.argmax(prominences)]

        peak_position = spatial[peak_idx]
        peak_height = values[peak_idx]

        return peak_position, peak_height, peak_idx

    def _compute_fwhm(
        self, spatial: np.ndarray, values: np.ndarray, peak_idx: int
    ) -> Optional[float]:
        """
        Compute Full Width at Half Maximum (FWHM).

        Args:
            spatial: Spatial coordinates
            values: Dose/LET values
            peak_idx: Index of the peak

        Returns:
            float or None: FWHM value, or None if cannot be computed
        """
        peak_height = values[peak_idx]
        half_max = peak_height / 2.0

        # Find points where curve crosses half maximum
        # Look for left crossing (before peak)
        left_idx = None
        for i in range(peak_idx, 0, -1):
            if values[i] <= half_max:
                left_idx = i
                break

        # Look for right crossing (after peak)
        right_idx = None
        for i in range(peak_idx, len(values)):
            if values[i] <= half_max:
                right_idx = i
                break

        if left_idx is None or right_idx is None:
            return None

        # Linear interpolation for more accurate crossing points
        if left_idx < len(values) - 1:
            x_left = np.interp(
                half_max,
                [values[left_idx], values[left_idx + 1]],
                [spatial[left_idx], spatial[left_idx + 1]],
            )
        else:
            x_left = spatial[left_idx]

        if right_idx > 0:
            x_right = np.interp(
                half_max,
                [values[right_idx - 1], values[right_idx]],
                [spatial[right_idx - 1], spatial[right_idx]],
            )
        else:
            x_right = spatial[right_idx]

        fwhm = abs(x_right - x_left)
        return fwhm

    def _extract_distal_falloff(
        self,
        spatial: np.ndarray,
        values: np.ndarray,
        peak_idx: int,
        peak_height: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the distal falloff region beyond the peak.

        Args:
            spatial: Spatial coordinates
            values: Dose/LET values
            peak_idx: Index of the peak
            peak_height: Height of the peak

        Returns:
            tuple: (falloff_spatial, falloff_values) arrays
        """
        # Define falloff region: from peak to end
        falloff_spatial = spatial[peak_idx:]
        falloff_values = values[peak_idx:]

        # Normalize by peak height for comparison
        falloff_values_norm = falloff_values / peak_height

        return falloff_spatial, falloff_values_norm

    def _compare_falloff_curves(
        self,
        spatial_true: np.ndarray,
        values_true: np.ndarray,
        spatial_pred: np.ndarray,
        values_pred: np.ndarray,
    ) -> float:
        """
        Compare distal falloff curves using interpolation.

        Args:
            spatial_true: True falloff spatial points
            values_true: True falloff values (normalized)
            spatial_pred: Predicted falloff spatial points
            values_pred: Predicted falloff values (normalized)

        Returns:
            float: MAE of falloff curves
        """
        if len(spatial_true) < 2 or len(spatial_pred) < 2:
            return np.nan

        # Find common spatial range
        spatial_min = max(spatial_true[0], spatial_pred[0])
        spatial_max = min(spatial_true[-1], spatial_pred[-1])

        if spatial_min >= spatial_max:
            return np.nan

        # Create common spatial grid for comparison
        n_points = min(len(spatial_true), len(spatial_pred), 100)
        common_spatial = np.linspace(spatial_min, spatial_max, n_points)

        # Interpolate both curves to common grid
        try:
            interp_true = interp1d(
                spatial_true,
                values_true,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interp_pred = interp1d(
                spatial_pred,
                values_pred,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            values_true_interp = interp_true(common_spatial)
            values_pred_interp = interp_pred(common_spatial)

            # Compute MAE
            mae = np.mean(np.abs(values_true_interp - values_pred_interp))
            return mae

        except Exception:
            return np.nan

    def compute_peak_position_error(
        self,
        data_true: pd.DataFrame,
        data_pred: pd.DataFrame,
        feature: str,
    ) -> float:
        """
        Compute peak position error for a feature.

        Args:
            data_true: Ground truth DataFrame with spatial coord and
                feature
            data_pred: Predicted DataFrame with spatial coord and
                feature
            feature: Name of the feature column to analyze

        Returns:
            float: Absolute difference in peak positions (spatial units,
                typically mm)
        """
        spatial_true = data_true[self.spatial_coordinate].values
        y_true = data_true[feature].values
        spatial_pred = data_pred[self.spatial_coordinate].values
        y_pred = data_pred[feature].values

        self._validate_inputs(spatial_true, y_true, spatial_pred, y_pred)

        peak_pos_true, _, _ = self._find_peak_position(spatial_true, y_true)
        peak_pos_pred, _, _ = self._find_peak_position(spatial_pred, y_pred)

        return abs(peak_pos_true - peak_pos_pred)

    def compute_peak_height_error(
        self,
        data_true: pd.DataFrame,
        data_pred: pd.DataFrame,
        feature: str,
    ) -> float:
        """
        Compute peak height error for a feature.

        Args:
            data_true: Ground truth DataFrame
            data_pred: Predicted DataFrame
            feature: Name of the feature column

        Returns:
            float: Absolute difference in peak heights (feature units,
                e.g., MeV/mm for LET)
        """
        spatial_true = data_true[self.spatial_coordinate].values
        y_true = data_true[feature].values
        spatial_pred = data_pred[self.spatial_coordinate].values
        y_pred = data_pred[feature].values

        self._validate_inputs(spatial_true, y_true, spatial_pred, y_pred)

        _, peak_height_true, _ = self._find_peak_position(spatial_true, y_true)
        _, peak_height_pred, _ = self._find_peak_position(spatial_pred, y_pred)

        return abs(peak_height_true - peak_height_pred)

    def compute_fwhm_error(
        self,
        data_true: pd.DataFrame,
        data_pred: pd.DataFrame,
        feature: str,
    ) -> Optional[float]:
        """
        Compute FWHM error for a feature.

        Args:
            data_true: Ground truth DataFrame
            data_pred: Predicted DataFrame
            feature: Name of the feature column

        Returns:
            float or None: Absolute difference in FWHM (spatial units,
                typically mm), or None if cannot be computed
        """
        spatial_true = data_true[self.spatial_coordinate].values
        y_true = data_true[feature].values
        spatial_pred = data_pred[self.spatial_coordinate].values
        y_pred = data_pred[feature].values

        self._validate_inputs(spatial_true, y_true, spatial_pred, y_pred)

        _, _, peak_idx_true = self._find_peak_position(spatial_true, y_true)
        _, _, peak_idx_pred = self._find_peak_position(spatial_pred, y_pred)

        fwhm_true = self._compute_fwhm(spatial_true, y_true, peak_idx_true)
        fwhm_pred = self._compute_fwhm(spatial_pred, y_pred, peak_idx_pred)

        if fwhm_true is None or fwhm_pred is None:
            return None

        return abs(fwhm_true - fwhm_pred)

    def compute_distal_falloff_error(
        self,
        data_true: pd.DataFrame,
        data_pred: pd.DataFrame,
        feature: str,
    ) -> float:
        """
        Compute distal falloff error for a feature.

        Args:
            data_true: Ground truth DataFrame
            data_pred: Predicted DataFrame
            feature: Name of the feature column

        Returns:
            float: MAE of normalized distal falloff curves
                (dimensionless, 0-1 scale)
        """
        spatial_true = data_true[self.spatial_coordinate].values
        y_true = data_true[feature].values
        spatial_pred = data_pred[self.spatial_coordinate].values
        y_pred = data_pred[feature].values

        self._validate_inputs(spatial_true, y_true, spatial_pred, y_pred)

        _, peak_height_true, peak_idx_true = self._find_peak_position(
            spatial_true, y_true
        )
        _, peak_height_pred, peak_idx_pred = self._find_peak_position(
            spatial_pred, y_pred
        )

        falloff_spatial_true, falloff_values_true = self._extract_distal_falloff(
            spatial_true, y_true, peak_idx_true, peak_height_true
        )
        falloff_spatial_pred, falloff_values_pred = self._extract_distal_falloff(
            spatial_pred, y_pred, peak_idx_pred, peak_height_pred
        )

        mae = self._compare_falloff_curves(
            falloff_spatial_true,
            falloff_values_true,
            falloff_spatial_pred,
            falloff_values_pred,
        )

        return mae

    def compute_all(
        self,
        data_true: pd.DataFrame,
        data_pred: pd.DataFrame,
        features: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compute all Bragg peak metrics for specified features.

        Args:
            data_true: Ground truth DataFrame with spatial coordinate
                and feature columns
            data_pred: Predicted DataFrame with spatial coordinate and
                feature columns
            features: List of feature column names to analyze. If None,
                analyzes all columns except spatial coordinate.

        Returns:
            pd.DataFrame: Metrics for each feature with columns:
                - Peak Position Error (mm)
                - Peak Height Error (feature units)
                - FWHM Error (mm)
                - Distal Falloff Error (dimensionless)
        """
        if features is None:
            features = [
                col for col in data_true.columns if col != self.spatial_coordinate
            ]

        results = []
        for feature in features:
            if feature not in data_pred.columns:
                continue

            try:
                peak_pos_err = self.compute_peak_position_error(
                    data_true, data_pred, feature
                )
                peak_height_err = self.compute_peak_height_error(
                    data_true, data_pred, feature
                )
                fwhm_err = self.compute_fwhm_error(data_true, data_pred, feature)
                falloff_err = self.compute_distal_falloff_error(
                    data_true, data_pred, feature
                )

                results.append(
                    {
                        "Feature": feature,
                        "Peak Position Error (mm)": peak_pos_err,
                        "Peak Height Error (a.u.)": peak_height_err,
                        "FWHM Error (mm)": fwhm_err,
                        "Distal Falloff Error": falloff_err,
                    }
                )
            except Exception as e:
                # Log warning but continue with other features
                print(f"Warning: Could not compute metrics for {feature}: {e}")
                continue

        return pd.DataFrame(results)
