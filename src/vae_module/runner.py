import time

import numpy as _np
import torch


def train_step(
    model, 
    optimizer, 
    trainloader, 
    device, 
    loss_fn
):
    """
    Perform one training step for the VAE model.

    Args:
        - model (nn.Module): VAE model to be trained.
        - optimizer (torch.optim.Optimizer): Optimizer for model parameter
          updates. 
        - trainloader (DataLoader): DataLoader for the training dataset.
        - device (torch.device): Device on which to perform computations ('cpu'
          or 'cuda'). 
        - loss_fn (callable): Loss function for calculating the training loss.

    Returns:
        tuple: A tuple containing the following elements:
            - average_train_loss_per_sample (float): Average training loss per
              sample for the epoch. 
            - train_loss_mse_per_sample (float): Average MSE loss per sample for
              the epoch. 
            - train_loss_kld_per_sample (float): Average KLD loss per sample for
              the epoch. 
            - epoch_time (float): Time taken for the epoch in seconds.
    """
    model.train()
    # train_loss, train_loss_mse, train_loss_kld = 0, 0, 0
    # num_samples = 0  # Total number of samples processed
    
    mse_loss = 1.0 if not model.history['train_loss_mse'] else model.history[
        'train_loss_mse'][-1]
    print("MSE loss at training start: ", mse_loss)
    
    gamma = model.gamma
    print("gamma at training start: ", gamma)
    
    half_log_two_pi = 0.91893
    
    epoch_loss = 0
    
    # Record the start time of the epoch
    start_time = time.time()
    
    for batch_idx, x in enumerate(trainloader):
        print("training batch ", batch_idx)
        
        x = x.to(device)
        
        # for each mini-batch, explicitly set the gradients to zero before
        # starting the backpropagation, as PyTorch by default accumulates the
        # gradients on subsequent backward passes 
        optimizer.zero_grad()
        
        # get reconstructed data
        x_recon, mu, logvar = model(x)
        
        # MSE loss between original data and reconstructed ones for current batch
        b_mse_loss = ((x - x_recon)**2).sum() / x.numel()
        print("MSE loss for current batch: ", b_mse_loss.item())
        
        # KL divergence between encoder distrib. and N(0,1) distrib.
        kl_loss = model.encoder.kl / mu.numel()
        print("KL loss for current batch: ", kl_loss.item())
        
        # 'variance law': this is a sanity check to test if the regularization
        # effect of the KL-divergence is properly working
        square_sigma = (mu ** 2 + torch.exp(logvar)).sum() / mu.numel()
        print("square_sigma (var law check): ", square_sigma.item())
        
        # build the loss
        gen_loss = (0.5 * ((x - x_recon) / gamma)**2 + 
                    _np.log(gamma) + half_log_two_pi).sum() / x.numel()
        print("gen_loss: ", gen_loss.item())
        
        loss = kl_loss + gen_loss
        print("total loss: ", loss.item())
        epoch_loss += loss
        
        # Call the loss function
        # total_loss, loss_MSE, loss_KLD = loss_fn(recon_batch, data, mu, logvar)
        
        # Perform backward pass on the loss component
        loss.backward()
        
        mse_loss = min(mse_loss, mse_loss*.99 + b_mse_loss.item()*.01)       
        gamma = _np.sqrt(mse_loss)
        
        """
        train_loss += total_loss.item()
        train_loss_mse += loss_MSE.item() 
        train_loss_kld += loss_KLD.item()
        num_samples += len(data)
        """
        
        # Call optimizer.step() to update parameters
        optimizer.step()
        print("--------------------------------")
    
    # Calculate and append the average training loss per batch
    epoch_loss /= len(trainloader)
    
    model.update_gamma(gamma)
    print("Updating gamma with value: ", gamma)
    
    print("ending training epoch...")
    print("===========================")
    print('')
    
    # Record and append the train time of the epoch
    end_time = time.time()
           
    return (
        epoch_loss.item(), # mean epoch loss per batch
        mse_loss,   # mse loss for the last batch in current epoch
        kl_loss.item(),    # kl loss for the last batch in current epoch
        end_time - start_time # training time of the current epoch
    )

def validate_step(
    model, 
    valloader, 
    loss_fn, 
    device
):
    """
    Perform one validation step for the VAE model.

    Args:
        - model (nn.Module): VAE model to be evaluated.
        - valloader (DataLoader): DataLoader for the validation dataset.
        - device (torch.device): Device on which to perform computations ('cpu'
          or 'cuda'). 
        - loss_fn (callable): Loss function for calculating the validation loss.

    Returns:
        float: Average validation loss per sample for the epoch.
    """
    model.eval()
    val_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for data in valloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            
            # Call the loss function
            total_loss, _, _ = loss_fn(recon_batch, data, mu, logvar)
            val_loss += total_loss.item()
            num_samples += len(data)
    
    # Calculate the average loss per sample
    average_val_loss = val_loss / num_samples
    
    return average_val_loss

def get_reconstructed_data_and_loss(
    model, 
    dataloader, 
    device, 
    loss_fn
):
    """
    Get the reconstructed data and loss from a DataLoader object using a model.

    Args:
        - model (nn.Module): VAE model used for reconstruction.
        - dataloader (DataLoader): DataLoader object containing the data.
        - device (torch.device): Device on which to perform computations ('cpu'
          or 'cuda'). 
        - loss_fn: The loss function used for reconstruction.

    Returns:
        tuple: A tuple containing two elements:
            - Tensor: Reconstructed data.
            - float: Average reconstruction loss per sample.
    """
    reconstructed_data = []
    total_loss = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            print(recon_batch)
            loss, _, _ = loss_fn(recon_batch, data, mu, logvar)
            total_loss += loss.item()
            num_samples += len(data)
            reconstructed_data.append(recon_batch.cpu())
    
    avg_loss_per_sample = total_loss / num_samples
    reconstructed_data = torch.cat(reconstructed_data, dim=0)
    return reconstructed_data, avg_loss_per_sample
