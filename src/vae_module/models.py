import sys as _sys
import time

_sys.path.append("../../src")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoenc.helpers import concat_lin_layers, concat_rev_lin_layers

from .runner import train_step, validate_step


class customLoss(nn.Module):
    """
    Custom loss function for the Variational Autoencoder.

    The Variational Autoencoder (VAE) requires a loss function that combines a
    reconstruction loss (typically Mean Squared Error) and a Kullback-Leibler
    Divergence (KLD) loss to ensure that the latent space distribution closely
    matches a prior distribution. 

    Args:
        beta (float, optional): Weight parameter for the KLD loss. Default is
        1.0. 

    Attributes:
        mse_loss (torch.nn.MSELoss): Mean Squared Error loss function.
        beta (float): Weight parameter for the KLD loss.

    Methods:
        forward(x_recon, x, mu, logvar): Computes the total loss, Mean Squared
        Error (MSE) loss, and Kullback-Leibler Divergence (KLD) loss. 
    """
    def __init__(self, beta=1.0):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.beta = beta
        
    def forward(self, x_recon, x, mu, logvar):
        """
        Calculate the total loss, Mean Squared Error (MSE) loss, and
        Kullback-Leibler Divergence (KLD) loss. 

        Args:
            - x_recon (torch.Tensor): Reconstructed data from the VAE.
            - x (torch.Tensor): Original input data batch. 
            - mu (torch.Tensor): Mean of the latent space distribution.
            - logvar (torch.Tensor): Log variance of the latent 
              space distribution. 

        Returns:
            tuple: A tuple containing the total loss, MSE loss, and KLD loss.

        """
        loss_MSE = self.mse_loss(x_recon, x)
        
        loss_KLD = -0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar)).mean()
        
        total_loss = loss_MSE + self.beta * loss_KLD
        
        return total_loss, loss_MSE, loss_KLD
    
def weights_init_uniform_rule(m):
    """
    Initialize the weights of a neural network module using a uniform
    distribution and a rule-based approach. 

    Args:
        m (nn.Module): The neural network module to initialize.

    Returns:
        None: The weights of the module are initialized in-place.

    Notes:
        This function initializes the weights of linear (fully connected) layers
        in the neural network module `m`. 
        For each linear layer, the weights are initialized from a uniform
        distribution with a scaling factor computed based on the number of input
        features to the layer. Biases are initialized to zeros. 
    """
    classname = m.__class__.__name__
    # for every Linear layer in a model...
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        
        # compute a scaling factor
        y = 1.0 / np.sqrt(n)
        
        # initialize the weights
        m.weight.data.uniform_(-y, y)
        
        # initialize the biases of the linear layer with zeros
        m.bias.data.fill_(0)

class VaeModular(nn.Module):
    def __init__(
        self, input_size:int, 
        hidden_layers_size:list, 
        latent_dim:int=3, 
        normalization_layer=None,
        activation_layer=nn.ReLU,
        output_activation_layer=nn.ReLU
    ):
        """
        Variational Autoencoder (VAE) with a modular architecture.

        Args:
            input_size (int): Input dimension.
            hidden_layers_size (list): List of hidden layer sizes.
            latent_dim (int, optional): Size of the latent space. Defaults to 3.
            batch_norm (bool, optional): Whether to apply batch normalization.
            Defaults to True. 
        """
        super(VaeModular, self).__init__()
        
        # Use the * operator to unpack the list of layers into individual
        # arguments for nn.Sequential       
        
        # Encoder
        self.encoder = nn.Sequential(
            *concat_lin_layers(
                input_shape=input_size,
                hidden_nodes=hidden_layers_size,
                normalization=normalization_layer,
                activation=activation_layer
            )
        )
        self.out_features_ = self.encoder[-1][0].out_features
        
        """
        # Latent vectors mu and logvar
        self.fc_latent = nn.Linear(self.out_features_, latent_dim)
        self.bn_latent = normalization_layer(num_features=latent_dim)
        
        # Layer `fc_latent` is responsible for transforming the output of the
        # encoder layers into the latent space
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        """
        self.fc_mu = nn.Linear(self.out_features_, latent_dim)
        self.fc_logvar = nn.Linear(self.out_features_, latent_dim)
        
        # Sampling vector
        # self.fc_z = nn.Linear(latent_dim, latent_dim)
        # self.bn_z = normalization_layer(latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.out_features_)
        self.bn_z = normalization_layer(self.out_features_)
        
        # Decoder
        # self.latent_to_output = nn.Linear(latent_dim, self.out_features_)
        # self.bn_lto = normalization_layer(self.out_features_)
        
        self.decoder = nn.Sequential(
            *concat_rev_lin_layers(
                output_shape=input_size,
                hidden_nodes=hidden_layers_size,
                normalization=normalization_layer,
                activation=activation_layer,
                output_activation=output_activation_layer
            )
        )
        
        self.history = None
        
    def encode(self, x):
        """
        Encodes input data into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Tuple containing mu and logvar.
        """
        # bn_latent = F.relu(self.bn_latent(self.fc_latent(self.encoder(x))))
        encoder_output = self.encoder(x)
        
        # mu = self.fc_mu(bn_latent)
        mu = self.fc_mu(encoder_output)
        
        # We usually don't directly compute the stddev but the log of the
        # squared stddev instead, which is the variance. The reasoning is
        # similar to why we use softmax, instead of directly outputting numbers
        # in fixed range [0, 1], the network can output a wider range of numbers
        # which we can later compress down. 
        # logvar = self.fc_logvar(bn_latent)
        logvar = self.fc_logvar(encoder_output)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log variance of the latent space. 

        Returns:
            torch.Tensor: Reparameterized latent space.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            
            # contains_nan = torch.isnan(std).any()
            # if contains_nan:
            #     print("Tensor contains NaN values.")
            
            eps = torch.randn_like(std)
            # 'torch.randn_like' returns a tensor with the same size as input
            # that is filled with random numbers from a normal distribution with
            # mean 0 and variance 1.
            
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu
        
    def decode(self, z):
        """
        Decodes the latent space back into the original input space.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            torch.Tensor: Reconstructed input data.
        """
        fc_z = F.relu(self.bn_z(self.fc_z(z)))
        
        # latent_to_output = F.relu(self.bn_lto(self.latent_to_output(fc_z)))
        
        # return self.decoder(latent_to_output)
        return self.decoder(fc_z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Tuple containing the reconstructed input data, mu, and logvar.
        """
        # Encoding
        mu, logvar = self.encode(x)         
        
        # Reparameterization
        z = self.reparameterize(mu, logvar) 
        
        # Decoding
        recon_x = self.decode(z)       
        
        return recon_x, mu, logvar
    
    def fit(
        self, 
        trainloader, 
        optimizer, 
        loss_fn, 
        device, 
        num_epochs=50, 
        valloader=None, 
        verbose=False, 
        show_every=50
    ):
        """
        Train the VAE model using the provided data.

        Args:
            - trainloader (DataLoader): DataLoader for the training data.
            - optimizer (torch.optim.Optimizer): Optimizer for updating model
              parameters. 
            - loss_fn: Loss function for computing the training loss.
            - device (torch.device): Device to be used for training (e.g.,
              'cuda' or 'cpu'). 
            - num_epochs (int, optional): Number of epochs for training. Default
              is 50. 
            - valloader (DataLoader, optional): DataLoader for the validation
              data. Default is None. 
            - verbose (bool, optional): Whether to print training progress.
              Default is False. 
            - show_every (int, optional): Frequency of printing training
              progress. Default is 50. 
        
        Returns:
            None
            
        Note:
            The training progress can be printed to the console if 'verbose' is
            set to True.
            The 'history' property of the model will be updated with training
            metrics. 
            
        """
        if self.history is None:
            self.history = {
                'train_loss': [], 
                'train_loss_mse': [], 
                'train_loss_kld': [], 
                'train_times': [],
                'total_training_time': 0,
                'val_loss': []
            }
            
        # Record the start time of training
        start_training_time = time.time()
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # print(f"Epoch {epoch}/{num_epochs}:")
            
            # Training phase
            train_loss, train_loss_mse, train_loss_kld, train_time = train_step(
                self, optimizer, trainloader, device, loss_fn)
            
            # Append training metrics to history
            self.history['train_loss'].append(train_loss)
            self.history['train_loss_mse'].append(train_loss_mse)
            self.history['train_loss_kld'].append(train_loss_kld)
            self.history['train_times'].append(train_time)
            
            # If validation DataLoader is provided, calculate validation loss
            if valloader is not None:
                val_loss = validate_step(self, valloader, device)
                
                # Append validation loss to history
                self.history['val_loss'].append(val_loss)
            
            # Print or log the average training loss per sample for the epoch
            if verbose and epoch % show_every == 0:
                print(f'====> Epoch: {epoch}, Average training loss per '
                        'sample: {:.7f}'.format(train_loss))
                print(f'====> Epoch: {epoch}, Average MSE loss per sample: '
                        '{:.7f}'.format(train_loss_mse))
                print(f'====> Epoch: {epoch}, Average KLD loss per sample: '
                        '{:.7f}'.format(train_loss_kld))
                print(f'====> Epoch: {epoch}, Training duration: '
                        '{:.2f} seconds'.format(train_time))
                if valloader is not None:
                    print(f'====> Epoch: {epoch}, Average validation loss per '
                        'sample: {:.7f}'.format(val_loss))
        
        # Record the end time of training
        end_training_time = time.time()

        # Calculate the total training time
        total_training_time = end_training_time - start_training_time

        # Store the total training time in the history dictionary
        self.history['total_training_time'] = total_training_time
        
        
class Encoder(nn.Module):
    
    def __init__(
        self, 
        device,
        input_dim:int, 
        hidden_layers_dim:list, 
        latent_dim:int=3, 
        normalization_layer=None,
        activation_layer=nn.ReLU,
    ) -> None:
        super(Encoder, self).__init__()
        
        self.device = device
        
        # bottleneck dimensionality
        self.output_dim = latent_dim
        
        # variables deciding whether using the normalization and activation type
        self.use_norm = normalization_layer
        self.activation_layer = activation_layer
        
        # encoder layers hyperparameters
        self.input_dim = input_dim
        self.layers_dim = hidden_layers_dim
        
        # Use the * operator to unpack the list of layers into individual
        # arguments for nn.Sequential       
        
        # Encoder
        self.encoder = nn.Sequential(
            *concat_lin_layers(
                input_shape=self.input_dim,
                hidden_nodes=self.layers_dim,
                normalization=self.use_norm,
                activation=self.activation_layer
            )
        )
        
        # Adding mean and logvar projections
        self.mu = nn.Linear(self.layers_dim[-1], self.output_dim)
        self.logvar = nn.Linear(self.layers_dim[-1], self.output_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        
        mu_z = self.mu(x)
        logvar_z = self.logvar(x)
        
        return mu_z, logvar_z
            
            
class Decoder(nn.Module):
    def __init__(
        self, 
        output_dim:int, # training data dim
        hidden_layers_dim:list, 
        input_dim:int=3, # latent space dim
        normalization_layer=None,
        activation_layer=nn.ReLU,
        output_activation_layer=nn.ReLU
    ) -> None:
        super(Decoder, self).__init__()
        
        # variables deciding whether using the normalization and activation type
        self.use_norm = normalization_layer
        self.activation_layer = activation_layer
        self.last_activation_layer = output_activation_layer
        
        # decoder layers hyperparameters
        self.input_dim = input_dim
        self.layers_dim = hidden_layers_dim
        self.output_dim = output_dim
        
        # In decoder, we first do fc project
        self.linear = nn.Linear(self.input_dim, self.layers_dim[-1])
        # no normalization ???
            
        self.decoder = nn.Sequential(
            *concat_lin_layers(
                input_shape=self.layers_dim[-1],
                hidden_nodes=self.layers_dim[::-1],
                normalization=self.use_norm,
                activation=self.activation_layer
            )
        )
        
        self.output = nn.Linear(self.layers_dim[0], self.output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.decoder(x)
        
        return self.output(x)
        
        
class AutoEncoder(nn.Module):
    
    def __init__(
        self,
        net_params: dict,
        optimizer_params: dict,
        device=None,
    ) -> None:
        super(AutoEncoder, self).__init__()
                
        # Autoencoder network parameters
        self.input_dim = net_params['input_dim']
        self.hidden_layers_dim = net_params['hidden_layers_dim']
        self.latent_dim = net_params['latent_dim']
        self.normalization = net_params['normalization']
        self.activation = net_params['activation']
        self.output_activation = net_params['output_activation']
        
        # Optimizer(s) parameters
        self.lr = optimizer_params['lr1'] # learning rate
        self.l2_reg = optimizer_params['l2_reg1'] # weight_decay (L2 penalty)
        
        self.device = device if not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize a 'target' normal distriobution for KL divergence
        self.norm = torch.distributions.Normal(0, 1)
        
        # build network architecture
        self.encoder = Encoder(
            input_dim=self.input_dim,
            device=self.device,
            hidden_layers_dim=self.hidden_layers_dim,
            latent_dim=self.latent_dim,
            normalization_layer=self.normalization,
            activation_layer=self.activation,
        ) 
        
        self.decoder = Decoder(
            output_dim=self.input_dim,
            hidden_layers_dim=self.hidden_layers_dim,
            input_dim=self.latent_dim,
            normalization_layer=self.normalization,
            activation_layer=self.activation,
            output_activation_layer=self.output_activation
        )
        
        # build the optimizer
        all_params = self.named_parameters()
        trainables = [par for name, par in all_params if par.requires_grad]
        
        self.opt = torch.optim.Adam(trainables, lr = self.lr,
                                    weight_decay=self.l2_reg)
        
        # tracking the 'gamma' parameter of Generative Loss (GL)
        self.gamma_x = 1.
        
        self.history = None
        
    def forward(self, x: torch.Tensor) -> tuple:
        mu_z, logvar_z = self.encoder(x)
        
        sigma_z = torch.exp(0.5 * logvar_z)
        
        # reparameterization trick
        z = mu_z + sigma_z * self.norm.sample(mu_z.shape).to(self.device)
        
        recon_x = self.decoder(z)
        
        return recon_x, mu_z, logvar_z
    
    def fit(
        self, 
        trainloader, 
        num_epochs=50, 
        valloader=None, 
        verbose=False, 
        show_every=50
    ):
        """
        Train the VAE model using the provided data.

        Args:
            - trainloader (DataLoader): DataLoader for the training data.
            - num_epochs (int, optional): Number of epochs for training. Default
            is 50. 
            - valloader (DataLoader, optional): DataLoader for the validation
            data. Default is None. 
            - verbose (bool, optional): Whether to print training progress.
            Default is False. 
            - show_every (int, optional): Frequency of printing training
            progress. Default is 50. 
        
        Returns:
            None
            
        Note:
            The training progress can be printed to the console if 'verbose' is
            set to True.
            The 'history' property of the model will be updated with training
            metrics. 
            
        """
        # Validation setup
        # best_val_loss = float('inf')
        # best_model_state_dict = None
        
        # History lists
        train_losses = []
        val_losses = []
        train_time_per_epoch = []
        gen_losses = []
        kl_losses = []
        var_losses = []
        gamma_x_list = []
        x_mse_losses_recon = []
            
        mse_loss = 1.
            
        # Record the start time of training
        start_training_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_time_start = time.time()
            epoch_loss = 0.
            
            for biter, x in enumerate(trainloader):
                x = x.to(self.device)
                
                loss, mse_loss = self.train_step(x, mse_loss)
                
                epoch_loss += loss
            
            epoch_loss /= (biter + 1)
            epoch_time_end = time.time()
            train_time = epoch_time_end - epoch_time_start
            
            train_losses.append(epoch_loss)
            train_time_per_epoch.append(train_time)
            gen_losses.append(self.gen_loss)
            kl_losses.append(self.kl_loss)
            var_losses.append(self.var_loss)
            gamma_x_list.append(self.gamma_x)
            x_mse_losses_recon.append(mse_loss)
            
            # If validation DataLoader is provided, calculate validation loss
            if valloader is not None:
                with torch.no_grad():
                    epoch_vloss = 0.
                    for b_iter, x in enumerate(valloader):
                       epoch_vloss += self.validate_step(x)
                    
                    epoch_vloss /= (b_iter + 1)
                                
                # Append validation loss to history
                val_losses.append(epoch_vloss)
                
                # if epoch_vloss < best_val_loss:
                #     best_val_loss = epoc_vloss
                #     best_model_state_dict = self.state_dict()
            else:
                epoch_vloss = None
            
            # Logging
            if verbose or (epoch + 1) % show_every == 0:
                self.log_epoch_progress(
                    epoch+1, num_epochs, epoch_loss, epoch_vloss, 
                    Gamma_x=self.gamma_x, recon_loss_mse=mse_loss,
                    var_loss=self.var_loss
                )
        
        # Record the end time of training
        end_training_time = time.time()

        # Calculate the total training time
        total_training_time = end_training_time - start_training_time
        
        print("Training [and Validation] time: {:.0f} min {:.2f} sec".format(
            total_training_time // 60, total_training_time % 60))

        # Return the history
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_time_per_epoch': train_time_per_epoch,
            'gen_losses': gen_losses,
            'kl_losses': kl_losses,
            'var_losses': var_losses,
            'gamma_x_list': gamma_x_list,
            'x_mse_losses_recon': x_mse_losses_recon,
        }    
        
    def train_step(self, x, mse_loss):
        self.train()
        
        # for each mini-batch, explicitly set the gradients to zero before
        # starting the backpropagation, as PyTorch by default accumulates the
        # gradients on subsequent backward passes 
        self.opt.zero_grad()
                           
        # loss calculation for current batch
        self.__build_loss(x)
            
        bloss = self.loss
            
        # Perform backward pass on the loss component
        bloss.backward()
            
        # Call optimizer.step() to update parameters
        self.opt.step()
            
        bmse_loss = self.mse_loss
            
        # calculate gamma value from mse losses combination
        mse_loss = min(mse_loss, mse_loss*.99 + bmse_loss.item()*.01)       
        self.gamma_x = np.sqrt(mse_loss)
            
        # Computing gamma according to the previous estimation of mse has
        # essentially the effect of keeping a constant balance between
        # reconstruction error and KL-divergence during the whole training: as
        # mse is decreasing, we normalize it in order to prevent a prevalence of
        # the KL-component, that would forbid further improvements of the
        # quality of reconstructions.
        
        # Return history components for the epoch
        return (
            bloss.item(), 
            mse_loss, 
        )
        
    def validate_step(self, x):
        self.eval()
        
        x = x.to(self.device)
                
        # loss calculation for current batch
        self.__build_loss(x)

        epoch_loss = self.loss
        
        return epoch_loss.item()
        
    def __build_loss(self, x: torch.Tensor):
        half_log_two_pi = np.log(2 * np.pi) / 2.
        
        recon_x, mu_z, logvar_z = self.forward(x)
        
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')        
        
        # KL divergence loss
        sigma_z = torch.exp(0.5 * logvar_z)
        kl_loss = 0.5 * torch.mean(mu_z ** 2 + sigma_z ** 2 - logvar_z - 1)
        
        # Calculate square of the standard deviation for the Gaussian prior
        var_loss = torch.mean(mu_z ** 2 + torch.exp(logvar_z))
        
        # Generalization loss
        gen_loss = torch.mean(0.5 * torch.square((x - recon_x) / self.gamma_x) +
                                np.log(self.gamma_x) + half_log_two_pi)
        
        # Total loss
        self.loss = kl_loss + gen_loss
        self.mse_loss = mse_loss
        self.kl_loss = kl_loss.item()
        self.gen_loss = gen_loss.item()
        self.var_loss = var_loss.item()
        
    def log_epoch_progress(self, 
        epoch, total_epochs, epoch_loss, epoch_vloss=None, stage=1, **kwargs):
        """
        Log the progress of the training or validation epoch.

        Args:
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs.
            epoch_loss (float): Training or validation loss for the epoch.
            epoch_vloss (float, optional): Validation loss for the epoch
            (default is None). 
            **kwargs: Additional keyword arguments for optional values to print.
        """
        print("Date: {date}".format(date=time.strftime('%Y-%m-%d %H:%M:%S')))
        print("Epoch [{}/{}]\t ".format(epoch, total_epochs), end='')
        if epoch_vloss is None:
            print("Loss: {:.4f}".format(epoch_loss))
        else:
            print("Loss: {:.4f}\t Validation Loss: {:.4f}".format(
                epoch_loss, epoch_vloss))
            
        for i, (key, value) in enumerate(kwargs.items(), start=1):
            print(f"\t{key}: {value:.6f}", end='')
            if i % 3 == 0:
                print() # Insert newline after every third item
        
        print('\n') # Add a newline at the end of the print statement
        
    def reconstruct(self, data_loader):
        self.eval() # Set the model to evaluation mode
        
        reconstruct_data = []
        mu_z_list = []
        
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device) # Move data to device
    
                # Forwarding the data
                recon_x, mu_z, _ = self.forward(x)
                                               
                # Append reconstructed data
                mu_z_list.append(mu_z.cpu().detach())
                reconstruct_data.append(recon_x.cpu().detach()) 
                
        return (
            torch.cat(reconstruct_data, dim=0), # Concatenate reconstructed data batches
            torch.cat(mu_z_list, dim=0), # Concatenate latent variable batches
        )