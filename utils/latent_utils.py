# src/utils/latent_utils.py

import logging
from typing import Any, Optional, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from configs.base_config import BaseVAEConfig
from utils.filesystem_utils import save_figure


# Define the interface expected by the mixin
@runtime_checkable
class LatentAware(Protocol):
    model: Any
    logger: logging.Logger
    config: BaseVAEConfig  
    color_column: Optional[str]
    latent_space_color_vals: Optional[np.ndarray]
    
    def extract_latents(self, X_loader, color_column: Optional[str] = None
                        ) -> np.ndarray:
        ...

    def plot_latent_space(self, X_latents: np.ndarray, method: str) -> Figure:
        ...

    def plot_and_save_latent_space(
        self, X_latents: np.ndarray, method: str, filename: str) -> None:
        ...


class LatentSpaceMixin:    
    def extract_latents(self: LatentAware, X_loader, color_column=None):
        if color_column is not None:
            self.color_column = color_column

        if self.model is None:
            self.logger.error("Model is not defined.")
            raise ValueError("Model must be set up before extracting latents")

        self.logger.info("Extracting latent space representation...")
        self.model.eval()
        latents = []
        color_values = []

        with torch.no_grad():
            for x in X_loader:
                x = x.to(self.model.device)
                if hasattr(self.model, 'identity_dim') and self.model.identity_dim > 0:
                    x_proc = x[:, :self.model.proc_dim]
                else:
                    x_proc = x
                mu, _ = self.model.encoder(x_proc)
                latents.append(mu.cpu().numpy())

                if color_column is not None:
                    try:
                        color_col_index = X_loader.col_mapping.get(color_column)
                        color_values.append(x[:, color_col_index].cpu().numpy())
                    except Exception as e:
                        raise ValueError(
                            f"Error extracting label from column {color_column}: {e}"
                            )

        latents = np.concatenate(latents, axis=0)
        if color_column is not None:
            self.latent_space_color_vals = np.concatenate(color_values, axis=0)

        return latents

    def plot_latent_space(
        self: LatentAware, X_latents: np.ndarray, method: str = 'pca'
    ) -> Figure:
        self.logger.info(f"Plotting latent space using method: {method.upper()}")

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=self.config.random_seed)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.config.random_seed)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=self.config.random_seed)
        else:
            raise ValueError(
                f"Unknown dimensionality reduction method '{method}'. "
                "Choose from: 'pca', 'tsne', or 'umap'."
            )

        reduced = reducer.fit_transform(X_latents)

        fig, ax = plt.subplots(figsize=(10, 10))
        if (hasattr(self, 'latent_space_color_vals') and 
            self.latent_space_color_vals is not None):
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1], marker='.', alpha=0.7,
                c=self.latent_space_color_vals, cmap='Spectral'
            )
            legend1 = ax.legend(
                *scatter.legend_elements(), 
                title=f"Feature '{self.color_column}'\nscaled values"
            )
            ax.add_artist(legend1)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], marker='.', alpha=1.0)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'Latent Space Variables ({method.upper()})')
        ax.axis('equal')
        ax.grid(True)

        return fig
    
    def plot_and_save_latent_space(
        self: LatentAware,
        X_latents: np.ndarray,
        method: str,
        filename: str,
        close_after_save: bool = False
    ) -> Figure:
        """
        Wrapper to plot the latent space and save the figure.

        Args:
            X_latents (np.ndarray): Latent features to visualize.
            method (str): Dimensionality reduction method ('pca', 'tsne', 'umap').
            filename (str): Output filename (without extension).
            close_after_save (bool): Whether to close figure after saving. 
                                   Defaults to False.
            
        Returns:
            Figure: The created matplotlib figure object.
        """
        fig = self.plot_latent_space(X_latents=X_latents, method=method)
        save_figure(fig, self.config.output_dir, filename, close_after_save=close_after_save)
        
        return fig
        
    def plot_and_save_multiple_latents(
        self: LatentAware,
        X_latents: np.ndarray,
        methods: list[str],
        close_after_save: bool = False
    ) -> list[Figure]:
        """ 
        Plot and save latent space visualizations using multiple methods.
        
        Args:
            X_latents (np.ndarray): Latent features to visualize.
            methods (list[str]): List of dimensionality reduction methods 
                                ('pca', 'tsne', 'umap').
            close_after_save (bool): Whether to close figures after saving. 
                                   Defaults to False.
        
        Returns:
            list[Figure]: List of created matplotlib figure objects.
        """
        if not methods:
            self.logger.warning("No methods provided for latent space plotting.")
            return []
        
        figures = []
        for method in methods:
            filename = f"latent_space_{method}"
            fig = self.plot_and_save_latent_space(X_latents, method, filename, close_after_save)
            figures.append(fig)
            
        return figures
