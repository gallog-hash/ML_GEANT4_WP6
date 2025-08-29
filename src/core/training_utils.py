# src/core/training_utils.py

from typing import Any, Dict

import torch
from torch import optim

from core.losses import CustomLossWithNegPenalty, InverseBetaLoss, SigmoidBetaLoss


def create_optimizer(params, optimizer_params: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = optimizer_params["learning_rate"]
    wd = optimizer_params["weight_decay"]
    optimizer = optimizer_params.get("optimizer", "Adam")

    if optimizer.lower() == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer.lower() == "sgd":
        momentum = optimizer_params.get("momentum", 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer}")

def build_loss_fn(loss_type: str, loss_params: Dict[str, Any]):
    if loss_type == "standard":
        return CustomLossWithNegPenalty(
            beta=loss_params.get("beta", 1.0),
            neg_penalty_weight=loss_params.get("neg_penalty_weight", 1.0),
            use_neg_penalty=loss_params.get("use_neg_penalty", False),
            scaler_mean=loss_params.get("scaler_mean", 0.0),
            scaler_scale=loss_params.get("scaler_scale", 1.0),
        )
    elif loss_type == "inverse":
        return InverseBetaLoss(
            beta_scale=loss_params.get("beta_scale", 1.0),
            neg_penalty_weight=loss_params.get("neg_penalty_weight", 1.0),
            use_neg_penalty=loss_params.get("use_neg_penalty", True),
            scaler_mean=loss_params.get("scaler_mean", 0.0),
            scaler_scale=loss_params.get("scaler_scale", 1.0),
            epsilon=loss_params.get("epsilon", 1e-6)
        )
    elif loss_type == "sigmoid":
        return SigmoidBetaLoss(
            beta_scale=loss_params.get("beta_scale", 10.0),
            recon_target=loss_params.get("recon_target", 0.01),
            neg_penalty_weight=loss_params.get("neg_penalty_weight", 1.0),
            use_neg_penalty=loss_params.get("use_neg_penalty", True),
            scaler_mean=loss_params.get("scaler_mean", 0.0),
            scaler_scale=loss_params.get("scaler_scale", 1.0)
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
