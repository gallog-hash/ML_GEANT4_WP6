# src/core/models/autoencoder.py

import time
from typing import Any, Dict, Optional, Tuple, Union

import optuna
import torch
import torch.nn as nn

from core.training_utils import build_loss_fn, create_optimizer
from utils import VAELogger

from .activations import ELUWithLearnableOffset
from .helpers import (
    concat_lin_layers,
    concat_rev_lin_layers,
    lin_layer,
    lin_layer_with_norm,
)

# --- Encoder and Decoder Classes ---

class Encoder(nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        hidden_layers_dim: list,
        latent_dim: int = 3,
        normalization_layer: Optional[Any] = None,
        activation_layer: Any = nn.ReLU,
        dropout_rate: float = 0.0,
    ) -> None:
        super(Encoder, self).__init__()
        self.device = device
        self.output_dim = latent_dim
        self.use_norm = normalization_layer
        self.activation_layer = activation_layer
        self.input_dim = input_dim
        self.layers_dim = hidden_layers_dim

        self.encoder = nn.Sequential(
            *concat_lin_layers(
                input_shape=input_dim,
                hidden_nodes=hidden_layers_dim,
                normalization=normalization_layer,
                activation=activation_layer,
                dropout_rate=dropout_rate,
            )
        )
        final_out = hidden_layers_dim[-1] if hidden_layers_dim else input_dim
        self.mu = nn.Linear(final_out, latent_dim)
        self.logvar = nn.Linear(final_out, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu_z = self.mu(x)
        logvar_z = self.logvar(x)
        return mu_z, logvar_z
            
class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_layers_dim: list,
        input_dim: int = 3,
        normalization_layer: Optional[Any] = None,
        activation_layer: Any = nn.ReLU,
        exit_activation_layer: Optional[Any] = None,
        use_exit_activation: bool = False,
        skip_norm_in_final: bool = False
    ) -> None:
        """
        Constructs the decoder network.

        If use_exit_activation is True, the decoder uses the full reversed
        architecture (including a final normalization and activation).
        Otherwise, it uses intermediate layers and a final plain linear layer.
        """
        super(Decoder, self).__init__()
        self.use_exit_activation = use_exit_activation
        self.input_dim = input_dim 
        self.hidden_layers_dim = hidden_layers_dim
        self.output_dim = output_dim 
        self.normalization_layer = normalization_layer
        self.activation_layer = activation_layer
        self.skip_norm_in_final = skip_norm_in_final
        if exit_activation_layer is None:
            exit_activation_layer = ELUWithLearnableOffset()
        self.exit_activation_layer = exit_activation_layer
        
        if self.use_exit_activation:
            layers = concat_rev_lin_layers(
                    input_shape=self.input_dim,
                    output_shape=self.output_dim,
                    hidden_nodes=self.hidden_layers_dim,
                    normalization=normalization_layer,
                    activation=activation_layer,
                    exit_activation=exit_activation_layer
            )
            # If skipping normalization on the final block, remove it:
            final_seq = layers[-1]
            if self.skip_norm_in_final and len(final_seq) >= 2 and \
               isinstance(final_seq[1], nn.BatchNorm1d):
                del layers[-1][1]
            self.decoder = nn.Sequential(*layers)
        else:
            # Build intermediate reversed layers and then add a plain linear
            # final layer.
            reversed_hidden = self.hidden_layers_dim[::-1]
            layers = []
            if normalization_layer is not None:
                layers.append(
                    lin_layer_with_norm(self.input_dim, reversed_hidden[0],
                                        normalization=normalization_layer,
                                        activation=activation_layer)
                )
                # Intermediate layers:
                for i in range(1, len(reversed_hidden)):
                    layers.append(
                        lin_layer_with_norm(reversed_hidden[i-1], 
                                            reversed_hidden[i],
                                            normalization=normalization_layer,
                                            activation=activation_layer)
                    )
            else:
                layers.append(
                    lin_layer(self.input_dim, reversed_hidden[0],
                              activation=activation_layer)
                )
                for i in range(1, len(reversed_hidden)):
                    layers.append(
                        lin_layer(reversed_hidden[i-1], reversed_hidden[i],
                                  activation=activation_layer)
                    )
            # Final layer: plain linear layer mapping to output_dim.
            layers.append(nn.Linear(reversed_hidden[-1], self.output_dim))
            self.decoder = nn.Sequential(*layers)

        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out   
    
# --- AutoEncoder Class ---

class AutoEncoder(nn.Module):
    def __init__(
        self, 
        architecture_params: dict, 
        opt: Optional[Union[torch.optim.Optimizer, Dict]] = None,
        loss_fn: Optional[Union[Any, dict]] = None, 
        logger_obj=None,
        device=None
    ) -> None:
        super(AutoEncoder, self).__init__()
        
        # Logger initialization
        if logger_obj is None:
            self.logger = VAELogger(
                name=self.__class__.__name__, log_level='debug').get_logger()
        else:
            self.logger = logger_obj.get_logger()
                
        # Autoencoder network parameters
        self.input_dim = architecture_params['input_dim']
        # Identity_dim specifies the number of identity features
        self.identity_dim = architecture_params.get('identity_dim', 0)
        # Processed dimension: feature to be encoded and decoded
        self.proc_dim = architecture_params['processed_dim']
        assert self.proc_dim >= 0, \
            "Processed dimension (input_dim - identity_dim) must be non-negative."
        
        self.hidden_layers_dim = architecture_params['hidden_layers_dim']
        self.latent_dim = architecture_params['latent_dim']
        self.normalization = architecture_params['normalization']
        self.activation = architecture_params['activation']
        self.dropout_rate = architecture_params.get('dropout_rate', 0.0)
        self.exit_activation_type = architecture_params['exit_activation_type']
        self.use_exit_activation = architecture_params.get('use_exit_activation', True)
        self.skip_norm_in_final = architecture_params.get('skip_norm_in_final', False)
        self.clamp_negatives = architecture_params.get('clamp_negatives', False)

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build encoder and decoder
        self.encoder = Encoder(
            input_dim=self.proc_dim,
            device=self.device,
            hidden_layers_dim=self.hidden_layers_dim,
            latent_dim=self.latent_dim,
            normalization_layer=self.normalization,
            activation_layer=self.activation,
            dropout_rate=self.dropout_rate,
        ) 
        
        self.decoder = Decoder(
            output_dim=self.proc_dim, # Output only the processed features
            hidden_layers_dim=self.hidden_layers_dim,
            input_dim=self.latent_dim,
            normalization_layer=self.normalization,
            activation_layer=self.activation,
            exit_activation_layer=self.exit_activation_type,
            use_exit_activation=self.use_exit_activation,
            skip_norm_in_final=self.skip_norm_in_final
        )
        
        self.opt = opt
        if isinstance(opt, dict):
            # If optimizer is a dictionary, create an instance of it.
            trainables = [
                par for _, par in self.named_parameters() if par.requires_grad
            ]
            self.opt = create_optimizer(trainables, opt)
        
        if self.opt is None:
            self.logger.warning("Optimizer not provided — training will fail "
                           "unless manually assigned.")
            
        self.loss_fn = loss_fn
        if isinstance(loss_fn, dict):
            # If loss function is a dictionary, build it.
            self.loss_fn = build_loss_fn(
                loss_type=loss_fn["type"],
                loss_params=loss_fn["params"]
            )
            
        if self.loss_fn is None:
            self.logger.warning("Loss function not provided — must be passed "
                           "to train_step/validate_step.")

        # Initialize a target normal distribution for KL divergence
        self.norm = torch.distributions.Normal(0, 1)
        
        # Tracking the 'gamma' parameter of Generative Loss (GL)
        self.gamma_x = 1.
        self.history = None
        
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Expected a torch.optim.Optimizer instance.")
        self.opt = optimizer
        self.logger.info("Optimizer assigned to model.")

    def set_loss_function(self, loss_fn: Any):
        self.loss_fn = loss_fn
        self.logger.info("Loss function assigned to model.")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor
                       ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If identity features are present, split the input accordingly.
        if self.identity_dim > 0:
            x_proc = x[:, :self.proc_dim]
            x_identity = x[:, self.proc_dim:]
        else:
            x_proc = x
            x_identity = None
        
        # Pass processed features through the encoder.
        mu, logvar = self.encoder(x_proc)
        # Reparameterization trick to sample from latent space.
        z = self.reparameterize(mu, logvar)
        # Decode the latent representation
        recon_proc = self.decoder(z)

        # Clamp negative values in processed features if enabled
        if self.clamp_negatives:
            recon_proc = torch.clamp(recon_proc, min=0.0)

        # If identity features exists, concatenate with the reconstructed part.
        if x_identity is not None:
            output = torch.cat([recon_proc, x_identity], dim=1)
        else:
            output = recon_proc

        return output, mu, logvar
        
    def fit(
        self, 
        trainloader, 
        num_epochs=50, 
        valloader=None, 
        verbose=False, 
        show_every=50,
        trial=None,
        val_loss_threshold=1e2,
    ) -> Dict[str, Any]:
        # Initialize history if not already set.
        if self.history is None:
            self.history = {
                'train_loss': [], 
                'train_loss_mse': [], 
                'train_loss_kld': [], 
                'train_neg_penalty': [],
                'train_times': [],
                'total_training_time': 0,
                'val_loss': []
            }
                       
        # Record the start time of training
        start_training_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_time_start = time.time()
            epoch_loss = 0.0
            epoch_loss_MSE = 0.0
            epoch_loss_KLD = 0.0
            epoch_neg_penalty = 0.0
            
            # Iterate over the training data loader
            for biter, x in enumerate(trainloader):
                # Move the input data to the appropriate device
                x = x.to(self.device)
                
                # Perform a training step 
                total_loss, loss_MSE, loss_KLD, neg_penalty = \
                    self.train_step(x)
                
                # Accumulate the loss for the current epoch
                epoch_loss += total_loss
                epoch_loss_MSE += loss_MSE
                epoch_loss_KLD += loss_KLD
                epoch_neg_penalty += neg_penalty
            
            num_batches = biter + 1
            epoch_loss /= num_batches
            epoch_loss_MSE /= num_batches
            epoch_loss_KLD /= num_batches
            epoch_neg_penalty /= num_batches
            train_time = time.time() - epoch_time_start
            
            # Save training metrics in history
            self.history['train_loss'].append(epoch_loss)
            self.history['train_loss_mse'].append(epoch_loss_MSE)
            self.history['train_loss_kld'].append(epoch_loss_KLD)
            self.history['train_neg_penalty'].append(epoch_neg_penalty)
            self.history['train_times'].append(train_time)
            
            # Validation phase
            if valloader is not None:
                with torch.no_grad():
                    epoch_vloss = 0.0
                    for b_iter, x in enumerate(valloader):
                        x = x.to(self.device)
                        total_loss = self.validate_step(x)
                        epoch_vloss += total_loss
                    epoch_vloss /= (b_iter + 1)
                    if epoch_vloss > val_loss_threshold:
                        epoch_vloss = float("inf")  # For Optuna to deprioritize this
                self.history['val_loss'].append(epoch_vloss)
                
            else:
                epoch_vloss = None

            # Logging progress if verbose.
            if verbose and ((epoch + 1) % show_every == 0 or epoch == 0):
                if epoch_vloss is not None:
                    self.logger.info(f"====> Epoch: {epoch+1}/{num_epochs}, " 
                                f"Training Loss: {epoch_loss:.7f}, "
                                f"Validation Loss: {epoch_vloss:.7f}, "
                                f"Negative Penalty: {epoch_neg_penalty:.7f}, "
                                f"Training Time: {train_time:.2f} sec")
                else:
                    self.logger.info(f"====> Epoch: {epoch+1}/{num_epochs}, "
                                f"Training Loss: {epoch_loss:.7f}, "
                                f"Training Time: {train_time:.2f} sec")
                    
            # Report to Optuna trial and check for pruning
            if trial is not None and epoch_vloss is not None:
                trial.report(epoch_vloss, epoch)
                if trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {epoch+1} with "
                                f"validation loss {epoch_vloss:.7f}")
                    raise optuna.TrialPruned()

        # Calculate the total training time
        total_training_time = time.time() - start_training_time
        self.history['total_training_time'] = total_training_time
        
        self.logger.info("Training [and Validation] time: "
                         "{:.0f} min {:.2f} sec".format(
            total_training_time // 60, total_training_time % 60))
        
        # Return the history
        return self.history    
        
    def train_step(
        self, 
        x: torch.Tensor, 
    ) -> Tuple[float, float, float, float]:
        assert self.opt is not None, "Optimizer not set. Use set_optimizer()."
        assert self.loss_fn is not None, "Loss function not set. Use set_loss_function()."

        self.train()  # Set the model to training mode
        self.opt.zero_grad()  # Zero the gradients
        x_recon, mu, logvar = self.forward(x)  # Forward pass
        
        assert self.loss_fn is not None, "Loss function is not set."
        
        total_loss, loss_components = self.loss_fn(x_recon, x, mu, logvar)
        total_loss.backward()  # Backward pass
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        
        # Update the model parameters
        self.opt.step()
        
        return (
            total_loss.item(), 
            loss_components["recon_loss"], # loss_MSE 
            loss_components["kl_loss"], 
            loss_components.get("neg_penalty", 0.0)
        )
        
    def validate_step(
        self, 
        x: torch.Tensor,
    ) -> float:
        self.eval()  # Set the model to evaluation mode

        # Move the input data to the appropriate device
        x = x.to(self.device)

        assert self.loss_fn is not None, "Loss function is not set."

        with torch.no_grad():
            x_recon, mu, logvar = self.forward(x)  # Forward pass
            # Compute the custom loss
            total_loss, _ = self.loss_fn(x_recon, x, mu, logvar)

        # Return the total loss
        return total_loss.item()
    
    def encode_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input into latent representation using the encoder.
        """
        return self.encoder(x)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vector into reconstructed output using the decoder.
        """
        return self.decoder(z)
    
    def reconstruct(self, data_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a full forward pass: encode input into latent space and decode
        back to reconstruction. 
        Also returns the latent means for each input.

        Args:
            data_loader (DataLoader): Input data in batches.

        Returns:
            Tuple[Tensor, Tensor]: (Reconstructed data, latent mean vectors)
        """
        self.eval() # Set the model to evaluation mode
        recon_data = []
        mu_list = []
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device) # Move data to device
                
                expected_dim = self.proc_dim + self.identity_dim
                assert x.shape[1] == expected_dim, (
                    f"[AutoEncoder.reconstruct] Expected input dim {expected_dim}, "
                    f"but got {x.shape[1]}."
                )
                
                # If identity features are present, split the input accordingly.
                has_identity = self.identity_dim > 0
                if has_identity:
                    x_proc = x[:, :self.proc_dim]
                    x_identity = x[:, self.proc_dim:]
                else:
                    x_proc = x
                    x_identity = None
    
                mu, _ = self.encode_latent(x_proc)
                x_rec = self.decode_latent(mu)

                # Clamp negative values in processed features if enabled
                if self.clamp_negatives:
                    x_rec = torch.clamp(x_rec, min=0.0)

                # If identity features exists, concatenate with the reconstructed part.
                if x_identity is not None:
                    output = torch.cat([x_rec, x_identity], dim=1)
                else:
                    output = x_rec
                                               
                # Append reconstructed data
                recon_data.append(output.cpu().detach())
                mu_list.append(mu.cpu().detach())
                
        return torch.cat(recon_data, dim=0), torch.cat(mu_list, dim=0)
