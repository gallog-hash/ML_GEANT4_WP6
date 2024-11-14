from typing import Optional

import torch.nn as nn


def lin_layer(
    ni:int, 
    no:int,
    activation: nn.Module =  nn.ReLU
) -> nn.Sequential:
    """
    Helper function to create a basic structure of a linear layer without
    BatchNormalization. 
    
    Parameters:
    - ni (int): Number of input features.
    - no (int): Number of output features.
    - activation (nn.Module, optional): Activation function to use after the
      linear layer. Defaults to nn.ReLU.
        
    Returns:
    - nn.Sequential: Sequential model containing a linear layer followed by
      the specified activation function.
    """
    return nn.Sequential(nn.Linear(ni, no), activation())

def lin_layer_with_norm(
    ni:int, 
    no:int,
    normalization: nn.Module = nn.BatchNorm1d,
    activation: nn.Module =  nn.ReLU
) -> nn.Sequential:
    """
    Helper function to create a basic structure of a linear layer with
    BatchNormalization applied. 
    
    Parameters:
    - ni (int): Number of input features.
    - no (int): Number of output features.
    - activation (nn.Module, optional): Activation function to use after the
      linear layer. Defaults to nn.ReLU.
    - normalization (nn.Module, optional): Normalization layer to use. Defaults
      to nn.BatchNorm1d.
        
    Returns:
    - nn.Sequential: Sequential model containing a linear layer,
      the specified normalization layer, and the specified activation function. 
    """
    return nn.Sequential(nn.Linear(ni, no), normalization(no), activation())

def concat_lin_layers(
    input_shape:int, 
    hidden_nodes:list, 
    normalization: Optional[nn.Module] = None,
    activation: nn.Module =  nn.ReLU
) -> list:
    """
    Concatenates linear layers with or without normalization applied.
    
    Parameters:
    - input_shape (int): Number of input features.
    - hidden_nodes (list): List of integers representing the number of nodes
      in hidden layers. 
    - normalization (nn.Module, optional): Normalization module to be applied
      after each linear layer. Defaults to nn.Identity (no normalization). 
    - activation (nn.Module): Activation function to use after each linear
      layer. Defaults to nn.ReLU.
        
    Returns:
    - list: List of Sequential models representing the concatenated linear
      layers. 
    """
        
    if normalization is nn.Identity:
        normalization = None
        
    output_shapes = [input_shape] + hidden_nodes
    if normalization is not None:
        layers = [
            lin_layer_with_norm(
                ni=output_shapes[i], 
                no=output_shapes[i+1],
                normalization=normalization,
                activation=activation
            )
            for i in range(len(output_shapes) - 1)
        ]
    else:
        layers = [
            lin_layer(
                ni=output_shapes[i], 
                no=output_shapes[i+1],
                activation=activation
            )
            for i in range(len(output_shapes) - 1)
        ]
        
    return layers
        
def concat_rev_lin_layers(
    output_shape:int, 
    hidden_nodes:list, 
    normalization: Optional[nn.Module] = nn.Identity,
    activation: nn.Module =  nn.ReLU,
    output_activation: nn.Module = nn.ReLU
):
    """
    Concatenates reversed linear layers with or without normalization applied. 
    
    Parameters:
    - input_shape (int): Number of input features.
    - hidden_nodes (list): List of integers representing the number of nodes
      in hidden layers. 
    - normalization (nn.Module, optional): Normalization module to be applied
      after each linear layer. Defaults to nn.Identity (no normalization). 
    - activation (nn.Module): Activation function to use after each linear
      layer. Defaults to nn.ReLU.
    - output_activation (nn.Module): Activation function to use for the output
      layer. Defaults to nn.RelU.
        
    Returns:
    - list: List of Sequential models representing the concatenated linear
      layers. 
    """
    
    if normalization is nn.Identity:
        normalization = None
        
    output_shapes = hidden_nodes[::-1] + [output_shape]
    layers = []
    if normalization is not None:
        for i in range(len(output_shapes) - 1):
            layers.append(
                lin_layer_with_norm(
                    ni=output_shapes[i], 
                    no=output_shapes[i+1],
                    normalization=normalization,
                    activation=activation if i != (len(output_shapes)-2) else output_activation
                )
            )
    else:
        for i in range(len(output_shapes) - 1):
            layers.append(
                lin_layer(
                    ni=output_shapes[i], 
                    no=output_shapes[i+1],
                    activation=activation if i != (len(output_shapes)-2) else output_activation
                )
            )

    return layers
    