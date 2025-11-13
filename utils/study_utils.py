# src/utils/study_utils.py
import optuna
from optuna.exceptions import DuplicatedStudyError

from utils.logger import VAELogger

logger = VAELogger("study_utils", "info").get_logger()

def load_optuna_study(study_name: str, storage: str) -> optuna.Study:
    logger.info(f"Loading Optuna study: '{study_name}' from storage: {storage}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.info("Study loaded successfully.")
        return study
    except DuplicatedStudyError:
        logger.error(f"Multiple studies named '{study_name}' found.")
        raise
    except Exception as e:
        logger.error(f"Failed to load Optuna study: {e}")
        raise

def summarize_best_trial(study, logger):
    if len(study.trials) == 0 or study.best_trial is None:
        logger.warning("No trials found in the study. Skipping summary.")
        return
    
    best_trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Best Trial Number: {best_trial.number}")
    logger.info(f"  Objective Value: {best_trial.value:.5f}")
    logger.info("  Parameters: ")
    
    complete_params = best_trial.params.copy()
    complete_params.update(best_trial.user_attrs)
    
    for key, val in complete_params.items():
        logger.info(f"   - {key}: {val}")