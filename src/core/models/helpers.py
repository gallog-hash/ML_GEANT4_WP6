# src/core/models/helpers.py

from typing import Callable, List, Optional

import torch.nn as nn


def lin_layer(
    ni:int, 
    no:int,
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """
    Create a basic linear layer with an activation function.
    
    Parameters:
    - ni (int): Number of input features.
    - no (int): Number of output features.
    - activation (Callable[[], nn.Module], optional): Activation function
      factory. Defaults to nn.ReLU.
    - dropout_rate (float, optional): Dropout rate. If > 0, a dropout layer is
      added after the linear layer. Defaults to 0.0 (no dropout).
        
    Returns:
    - nn.Sequential: A sequential module with a linear layer and the activation.
    """
    if dropout_rate > 0.0:
        return nn.Sequential(
            nn.Linear(ni, no),
            activation(),
            nn.Dropout(dropout_rate),
        )
    else:
        return nn.Sequential(nn.Linear(ni, no), activation())

def lin_layer_with_norm(
    ni:int, 
    no:int,
    normalization: Callable[[int], nn.Module] = nn.BatchNorm1d,
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """
    Create a linear layer followed by a normalization layer and an activation
    function. 
    
    Parameters:
    - ni (int): Number of input features.
    - no (int): Number of output features.
    - normalization (Callable[[int], nn.Module], optional): Normalization layer
      factory. Defaults to nn.BatchNorm1d.
    - activation (Callable[[int], nn.Module], optional): Activation function
      factory. Defaults to nn.ReLU.
    - dropout_rate (float, optional): Dropout rate. If > 0, a dropout layer is
      added after the linear layer. Defaults to 0.0 (no dropout).
        
    Returns:
    - nn.Sequential: A sequential module with a linear layer, normalization, and
      activation.  
    """
    if dropout_rate > 0.0:
        return nn.Sequential(
            nn.Linear(ni, no),
            normalization(no),
            nn.Dropout(dropout_rate),
            activation(),
        )
    else:
        return nn.Sequential(nn.Linear(ni, no), normalization(no), activation())

def concat_lin_layers(
    input_shape:int, 
    hidden_nodes:List[int], 
    normalization: Optional[Callable[[int], nn.Module]] = None,
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout_rate: float = 0.0 
) -> List[nn.Sequential]:
    """
    Concatenates multiple linear layers (with optional normalization and
    activation). 
    
    Parameters:
    - input_shape (int): Number of input features.
    - hidden_nodes (List[int]): List of integers representing the number of
      nodes in hidden layers. 
    - normalization (Optional[Callable[[int], nn.Module]], optional):
      Normalization layer factory. If provided (and not nn.Identity), applied
      after each linear layer. Defaults to None (no normalization).  
    - activation (nn.Module): (Callable[[], nn.Module], optional): Activation
      function factory. Defaults to nn.ReLU.
    - dropout_rate (float, optional): Dropout rate. Defaults to 0.0 (no dropout).
            
    Returns:
    - List[nn.Sequential]: A list of sequential modules representing the layers.
    """
    # Treat nn.Identity or None as no normalization.
    norm_func = None if (normalization == nn.Identity or normalization is None) \
        else normalization
        
    output_shapes = [input_shape] + hidden_nodes
    layers = []
    if normalization is not None:
        for i in range(len(output_shapes) - 1):
            if norm_func is not None:
                layers.append(lin_layer_with_norm(
                    ni=output_shapes[i],
                    no=output_shapes[i+1],
                    normalization=norm_func,
                    activation=activation,
                    dropout_rate=dropout_rate    
                ))
    else:
        layers.append(lin_layer(
            ni=output_shapes[i],
            no=output_shapes[i+1],
            activation=activation,
            dropout_rate=dropout_rate
        ))
    return layers
        
        
def concat_rev_lin_layers(
    input_shape: int, # latent_dim
    output_shape: int, # original input_dim 
    hidden_nodes: List[int],
    normalization: Optional[Callable[[int], nn.Module]] = None,
    activation: Callable[[], nn.Module] = nn.ReLU,
    exit_activation: Callable[[], nn.Module] = nn.Softplus
) -> List[nn.Sequential]:
    """
    Concatenate reversed linear layers with optional normalization and activation,
    ending with a final layer that maps to the output dimension. 
    
    Parameters:
    - input_shape (int): Number of input features (typically latent dimension).
    - output_shape (int): Number of output features (original input dimension).
    - hidden_nodes (List[int]): List of integers representing the number of
      nodes in hidden layers. 
    - normalization [Callable[[int], nn.Module]], optional): Normalization layer
      factory. If provided (and not nn.Identity), applied after each linear
      layer. Defaults to None (no normalization). 
    - activation (Callable[[], nn.Module], optional): Activation function
      factory. Defaults to nn.ReLU.
    - exit_activation (Callable[[], nn.Module], optional): Activation function
      factory for the final layer. Defaults to nn.Softplus.
        
    Returns:
    - List[nn.Sequential]: A list of sequential modules representing the
    reversed layers. 
    """
    # Treat nn.Identity or None as no normalization.
    norm_func = None if (normalization == nn.Identity or normalization is None) \
        else normalization
        
    output_shapes = [input_shape] + hidden_nodes[::-1]
    layers = []
    for i in range(len(output_shapes) - 1):
        if norm_func is not None:
            layers.append(lin_layer_with_norm(
                ni=output_shapes[i],
                no=output_shapes[i+1],
                normalization=norm_func,
                activation=activation,
            ))
        else:
            layers.append(lin_layer(
                ni=output_shapes[i],
                no=output_shapes[i+1],
                activation=activation,
            ))
    # Final layer: choose appropriate function based on normalization.
    if exit_activation is None:
    # Use a plain linear layer without any activation.
        final_layer = nn.Sequential(nn.Linear(output_shapes[-1], output_shape))
    else:
        if norm_func is not None:
            final_layer = lin_layer_with_norm(
                ni=output_shapes[-1],
                no=output_shape,
                normalization=norm_func,
                activation=exit_activation
            )
        else:
            final_layer = lin_layer(
                ni=output_shapes[-1],
                no=output_shape,
                activation=exit_activation
            )
    layers.append(final_layer)

    return layers
    