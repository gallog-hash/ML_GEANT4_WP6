import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional, Union

import numpy as np
import optuna
import optuna.visualization as vis
from optuna.exceptions import ExperimentalWarning

from configs.task_config import OptimizationAnalysisConfig
from core.base_pipeline import BaseVAEPipeline
from utils import display_plot
from utils import (
    load_config_from_json,
    load_optuna_study,
    save_figure,
    summarize_best_trial,
)


class VAEOptimizationAnalyzer(BaseVAEPipeline):
    def __init__(self, config: OptimizationAnalysisConfig):
        # Initialize BaseVAEPipeline (handles all path resolution, logging, etc.)
        super().__init__(config)

        # Set Optuna logging level
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        self.study = load_optuna_study(
            study_name=self.config.study_name, storage=self.config.study_path
        )
        summarize_best_trial(self.study, self.logger)

    def _create_capped_study(self, cap: float) -> optuna.Study:
        from optuna.trial import TrialState

        new_study = optuna.create_study(direction=self.study.direction)

        for trial in self.study.get_trials(deepcopy=True, states=[TrialState.COMPLETE]):
            if trial.value is not None:
                capped_value = min(trial.value, cap)
                new_study.add_trial(
                    optuna.trial.FrozenTrial(
                        number=trial.number,
                        trial_id=trial._trial_id,
                        value=capped_value,
                        values=None,
                        datetime_start=trial.datetime_start,
                        datetime_complete=trial.datetime_complete,
                        params=trial.params,
                        distributions=trial.distributions,
                        user_attrs=trial.user_attrs,
                        system_attrs=trial.system_attrs,
                        intermediate_values=trial.intermediate_values,
                        state=trial.state,
                    )
                )
        return new_study

    def plot_basic_study_analysis(self):
        self.logger.info(
            "Plotting study optimization history and parameter importance..."
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ExperimentalWarning)
                fig1 = vis.matplotlib.plot_optimization_history(self.study).figure
                if fig1 and fig1.axes:
                    fig1.axes[0].set_yscale("log")
                    display_plot(fig1)
                    save_figure(fig1, self.config.output_dir, "optimization_history")
                else:
                    self.logger.warning(
                        "plot_optimization_history returned no figure or axes."
                    )

                fig2 = vis.matplotlib.plot_param_importances(self.study).figure
                if fig2:
                    display_plot(fig2)
                    save_figure(fig2, self.config.output_dir, "param_importance")
                else:
                    self.logger.warning("plot_param_importances returned no figure.")
        except Exception as e:
            self.logger.info(f"Failed to generate matplotlib plots: {e}")

    def plot_timeline(self):
        self.logger.info("Plotting trial timeline with matplotlib backend...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ExperimentalWarning)
                fig = optuna.visualization.matplotlib.plot_timeline(self.study).figure
                display_plot(fig)
                save_figure(fig, self.config.output_dir, "trial_timeline")
        except Exception as e:
            self.logger.info(f"Failed to generate trial timeline plot: {e}")

    def plot_edf(self):
        self.logger.info("Plotting EDF with matplotlib backend...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ExperimentalWarning)
                fig = optuna.visualization.matplotlib.plot_edf(self.study).figure
                display_plot(fig)
                save_figure(fig, self.config.output_dir, "edf_plot")
        except Exception as e:
            self.logger.info(f"Failed to generate matplotlib EDF plot: {e}")

    def plot_parameters_slice(self, batch_size=3):
        self.logger.info(
            "Plotting slice plots of parameters in batches of 3 by importance..."
        )

        importances = optuna.importance.get_param_importances(self.study)
        sorted_params = [
            k for k, _ in sorted(importances.items(), key=lambda x: x[1], reverse=True)
        ]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ExperimentalWarning)
                for batch_start in range(0, len(sorted_params), batch_size):
                    batch_params = sorted_params[batch_start : batch_start + batch_size]

                    fig_axes = vis.matplotlib.plot_slice(
                        self.study, params=batch_params
                    )
                    if isinstance(fig_axes, np.ndarray):
                        fig = fig_axes[0].get_figure()
                    else:
                        fig = fig_axes.get_figure()

                    if fig and fig.axes:
                        fig.axes[0].set_yscale("log")
                        display_plot(fig)
                        save_figure(
                            fig,
                            self.config.output_dir,
                            f"slice_plot_{batch_start // batch_size}",
                        )
                    else:
                        self.logger.warning("plot_slice returned no figure or axes.")

        except Exception as e:
            self.logger.info(f"Failed to generate matlpotlib slice plots: {e}")

    def plot_contour(self, cap_z: bool = False):
        self.logger.info("Plotting contour plots of parameter pairs...")

        try:
            importances = optuna.importance.get_param_importances(self.study)
            sorted_params = [
                k
                for k, _ in sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                )
            ]

            z_cap = None
            if cap_z:
                values = [t.value for t in self.study.trials if t.value is not None]
                if values:
                    z_cap = float(np.quantile(values, 0.95))

            for _, (param1, param2) in enumerate(combinations(sorted_params, 2)):
                if cap_z and z_cap is not None:
                    study_to_use = self._create_capped_study(z_cap)
                else:
                    study_to_use = self.study
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ExperimentalWarning)
                    fig_ax = vis.matplotlib.plot_contour(
                        study_to_use, params=[param1, param2]
                    )
                if isinstance(fig_ax, np.ndarray):
                    fig = fig_ax[0].get_figure()
                else:
                    fig = fig_ax.get_figure()

                if fig:
                    display_plot(fig)
                    save_figure(
                        fig, self.config.output_dir, f"param_contour_{param1}_{param2}"
                    )
                else:
                    self.logger.warning("plot_contour returned no figure.")

            self.logger.info("Finished generating contour plots.")

        except Exception as e:
            self.logger.info(f"Failed to generate contour plots: {e}")


def main(config_path: Optional[Union[str, Path]] = None):
    # Load from CLI or fallback default
    if config_path is None:
        parser = argparse.ArgumentParser(description="VAE Optimization Analysis Script")
        parser.add_argument(
            "--config_path",
            type=str,
            default=str(
                Path(__file__).resolve().parent
                / "configs/optimization_analysis_config.json"
            ),
            help="Path to generation config JSON file",
        )
        args = parser.parse_args()
        config_path = args.config_path

    # Initialize configuration
    if config_path is None:
        raise ValueError("config_path cannot be None. Please provide a valid path.")
    try:
        config = load_config_from_json(
            config_path=Path(config_path), config_class=OptimizationAnalysisConfig
        )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Initialize the analyzer
    opt_analyzer = VAEOptimizationAnalyzer(config)

    # Plot optimization analysis
    opt_analyzer.plot_basic_study_analysis()
    opt_analyzer.plot_timeline()
    opt_analyzer.plot_edf()
    opt_analyzer.plot_parameters_slice()
    opt_analyzer.plot_contour(cap_z=True)


if __name__ == "__main__":
    if "get_ipython" in globals():
        # Running in interactive mode (e.g., VSCode Interactive Window or Jupyter)
        main("configs/optimization_analysis_config.json")
    else:
        # Standard command-line execution
        main()

