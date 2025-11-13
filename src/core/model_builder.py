# src/core/model_builder.py

from functools import partial

import torch.nn as nn

from .models import PELU, AutoEncoder, ELUWithLearnableOffset, ShiftedSoftplus


def build_vae_model_from_params(input_dim, n_processed_features, hyperparams, device):
    """
    Construct an AutoEncoder model using the provided architecture parameters.

    This function builds the VAE architecture only — it optionally configure the
    optimizer and does not configure the loss function, which should be assigned externally using `model.set_loss_function(...)`. 

    Args:
        input_dim (int): Number of raw input features.
        n_processed_features (int): Number of features after preprocessing
        (e.g., removal of identity features). 
        hyperparams (dict): Dictionary of architecture-related parameters. Must
        include: 
            - latent_dim
            - num_layers
            - hidden_layers_dim
            - use_dropout
            - dropout_rate
            - use_exit_activation (optional)
            - exit_activation_type (optional)
            - offset_init (optional)
        device (str): Device string ("cuda" or "cpu").

    Returns:
        AutoEncoder: Initialized AutoEncoder model (not compiled).
    """

    # Extract activation function
    activation_choice = hyperparams.get("exit_activation_type", None)
    output_activation = None  # Default: No output activation layer

    if activation_choice == "shifted_softplus":
        beta_softplus = hyperparams["beta_softplus"]
        output_activation = partial(ShiftedSoftplus, beta=beta_softplus)
    elif activation_choice == "elu_offset":
        offset_init = hyperparams["offset_init"]
        output_activation = partial(
            ELUWithLearnableOffset, offset_init=offset_init
        )
    elif activation_choice == "pelu":
        a_init = hyperparams["a_init"]
        b_init = hyperparams["b_init"]
        output_activation = partial(PELU, a_init=a_init, b_init=b_init)

    # Split hyperparameters into network and optimizer settings
    network_params = {
        "input_dim": input_dim,
        "identity_dim": input_dim - n_processed_features,
        "hidden_layers_dim": hyperparams["hidden_layers_dim"],
        "latent_dim": hyperparams["latent_dim"],
        "normalization": nn.BatchNorm1d,
        "activation": nn.ReLU,
        "exit_activation_type": output_activation,
        "use_exit_activation": hyperparams["use_exit_activation"],
        "skip_norm_in_final": hyperparams["skip_norm_in_final"],
        "processed_dim": n_processed_features,
        "use_dropout": hyperparams.get("use_dropout", False),
        "dropout_rate": hyperparams.get("dropout_rate", 0.0),
        "clamp_negatives": hyperparams.get("clamp_negatives", False),
    }

    # Define optimizer parameters only if corresponding keys exist in
    # hyperparams
    optimizer_keys = ["optimizer", "learning_rate", "weight_decay"]
    if all(key in hyperparams for key in optimizer_keys):
        optimizer_params = {
            "optimizer": hyperparams["optimizer"],
            "learning_rate": hyperparams["learning_rate"],
            "weight_decay": hyperparams["weight_decay"],
        }
    else:
        optimizer_params = None

    # Instantiate and return the AutoEncoder
    return AutoEncoder(
        architecture_params=network_params, 
        opt=optimizer_params, 
        device=device).to(device)
