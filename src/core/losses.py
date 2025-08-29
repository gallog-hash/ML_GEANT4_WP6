# src/core/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLossWithNegPenalty(nn.Module):
    """
    Custom loss function for the Variational Autoencoder with an additional
    penalty for negative outputs in the processed (non-identity) features.
    
    This loss is composed of:
      - A reconstruction loss (MSE).
      - A KL divergence loss.
      - An optional penalty term that increases the loss if any reconstructed
        output is negative. 
      
    Args:
      - beta (float, optional): Weight for the KL divergence loss (default: 1.0).
      - neg_penalty_weight (float, optional): Weight for the negative output
        penalty  (default: 1.0).
      - proc_dim (int, optional): Number of processed (non-identity) features.
        If provided, the MSE and negative penalty will be computed only on the
        processed portion.  
      - use_neg_penalty (bool, optional): Whether to apply the negative penalty.
        Defaults to True. 
    """
    def __init__(
        self,
        beta=1.0,
        neg_penalty_weight=1.0,
        proc_dim=None,
        use_neg_penalty=True,
        scaler_mean=None,
        scaler_scale=None
    ):
        super().__init__()
        self.beta = beta
        self.neg_penalty_weight = neg_penalty_weight
        self.proc_dim = proc_dim
        self.use_neg_penalty = use_neg_penalty

        self.scaled_zero = None
        if scaler_mean is not None and scaler_scale is not None:
            self.scaled_zero = -torch.tensor(
                scaler_mean, dtype=torch.float32
            ) / torch.tensor(scaler_scale, dtype=torch.float32)

    def forward(self, x_hat, x, mu, logvar):
        if self.proc_dim is not None:
            x_proc = x[:, :self.proc_dim]
            x_hat_proc = x_hat[:, :self.proc_dim]
        else:
            x_proc = x
            x_hat_proc = x_hat

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat_proc, x_proc, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        loss_components = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        # Penalty for outputs that would become negative after reverse-scaling
        if self.use_neg_penalty and self.scaled_zero is not None:
            if self.proc_dim is not None:
                threshold = self.scaled_zero[:self.proc_dim]
            else:
                threshold = self.scaled_zero
            threshold = threshold.to(x_hat.device)
            neg_penalty = torch.relu(threshold - x_hat_proc).mean()
            total_loss += self.neg_penalty_weight * neg_penalty
            loss_components["neg_penalty"] = neg_penalty.item()
        else:
            neg_penalty = 0.0

        return total_loss, loss_components

class InverseBetaLoss(nn.Module):
    def __init__(
        self,
        beta_scale=1.0,
        neg_penalty_weight=1.0,
        proc_dim=None,
        use_neg_penalty=True,
        scaler_mean=None,
        scaler_scale=None,
        epsilon=1e-6,
    ):
        super().__init__()
        self.beta_scale = beta_scale
        self.neg_penalty_weight = neg_penalty_weight
        self.proc_dim = proc_dim
        self.use_neg_penalty = use_neg_penalty
        self.epsilon = epsilon

        self.scaled_zero = None
        if scaler_mean is not None and scaler_scale is not None:
            self.scaled_zero = -torch.tensor(scaler_mean) / torch.tensor(scaler_scale)

    def forward(self, x_hat, x, mu, logvar):
        if self.proc_dim is not None:
            x_proc = x[:, :self.proc_dim]
            x_hat_proc = x_hat[:, :self.proc_dim]
            if self.scaled_zero is not None:
                threshold = self.scaled_zero[:self.proc_dim] 
            else:
                threshold = None
        else:
            x_proc = x
            x_hat_proc = x_hat
            threshold = self.scaled_zero

        recon_loss = F.mse_loss(x_hat_proc, x_proc, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        beta_eff = self.beta_scale / (recon_loss + self.epsilon)
        total_loss = recon_loss + beta_eff * kl_loss

        loss_components = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta_eff": beta_eff.item(),
        }

        if self.use_neg_penalty and threshold is not None:
            threshold = threshold.to(x_hat.device)
            neg_penalty = torch.relu(threshold - x_hat_proc).mean()
            total_loss += self.neg_penalty_weight * neg_penalty
            loss_components["neg_penalty"] = neg_penalty.item()
        else:
            neg_penalty = 0.0

        return total_loss, loss_components

class SigmoidBetaLoss(nn.Module):
    def __init__(
        self,
        beta_scale=10.0,
        recon_target=0.01,
        neg_penalty_weight=1.0,
        proc_dim=None,
        use_neg_penalty=True,
        scaler_mean=None,
        scaler_scale=None,
    ):
        super().__init__()
        self.beta_scale = beta_scale
        self.recon_target = recon_target
        self.neg_penalty_weight = neg_penalty_weight
        self.proc_dim = proc_dim
        self.use_neg_penalty = use_neg_penalty

        self.scaled_zero = None
        if scaler_mean is not None and scaler_scale is not None:
            self.scaled_zero = -torch.tensor(scaler_mean) / torch.tensor(scaler_scale)

    def forward(self, x_hat, x, mu, logvar):
        if self.proc_dim is not None:
            x_proc = x[:, :self.proc_dim]
            x_hat_proc = x_hat[:, :self.proc_dim]
            if self.scaled_zero is not None:
                threshold = self.scaled_zero[:self.proc_dim] 
            else:
                threshold = None
        else:
            x_proc = x
            x_hat_proc = x_hat
            threshold = self.scaled_zero

        recon_loss = F.mse_loss(x_hat_proc, x_proc, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Use sigmoid function to modulate beta
        beta_eff = torch.sigmoid(self.beta_scale * (recon_loss - self.recon_target))
        total_loss = recon_loss + beta_eff * kl_loss

        loss_components = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta_eff": beta_eff.item(),
        }

        if self.use_neg_penalty and threshold is not None:
            threshold = threshold.to(x_hat.device)
            neg_penalty = torch.relu(threshold - x_hat_proc).mean()
            total_loss += self.neg_penalty_weight * neg_penalty
            loss_components["neg_penalty"] = neg_penalty.item()
        else:
            neg_penalty = 0.0

        return total_loss, loss_components
