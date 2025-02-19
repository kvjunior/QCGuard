import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import networkx as nx
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F
from config import CONFIG
import logging
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.metrics import confusion_matrix

class QuantumMetricsPlotter:
    """Visualize quantum security metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_scale = 'Viridis'
    
    def plot_quantum_security(
        self,
        metrics: Dict[str, float],
        title: str = "Quantum Security Analysis"
    ) -> go.Figure:
        """Plot quantum security metrics"""
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        # Add radar plot
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Security Metrics'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=title
        )
        
        return fig
    
    def plot_security_evolution(
        self,
        metric_history: Dict[str, List[float]]
    ) -> go.Figure:
        """Plot security metrics evolution over time"""
        fig = go.Figure()
        
        for metric_name, values in metric_history.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric_name,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Security Metrics Evolution",
            xaxis_title="Training Step",
            yaxis_title="Security Level",
            yaxis=dict(range=[0, 1])
        )
        
        return fig

class Layer2Visualizer:
    """Visualize Layer-2 metrics and patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def plot_rollup_compression(
        self,
        original_data: torch.Tensor,
        compressed_data: torch.Tensor,
        title: str = "Layer-2 Compression Analysis"
    ) -> go.Figure:
        """Visualize rollup compression"""
        fig = make_subplots(rows=1, cols=2)
        
        # Original data distribution
        fig.add_trace(
            go.Histogram(
                x=original_data.flatten().numpy(),
                name="Original Data",
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # Compressed data distribution
        fig.add_trace(
            go.Histogram(
                x=compressed_data.flatten().numpy(),
                name="Compressed Data",
                nbinsx=50
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=True
        )
        
        return fig
    
    def plot_layer2_metrics(
        self,
        metrics: Dict[str, float]
    ) -> go.Figure:
        """Plot Layer-2 performance metrics"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values())
            )
        ])
        
        fig.update_layout(
            title="Layer-2 Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value"
        )
        
        return fig

class PrivacyVisualizer:
    """Enhanced privacy-preserving visualizations"""
    
    def __init__(
        self,
        epsilon: float = CONFIG.model.privacy_epsilon,
        min_cluster_size: int = 5
    ):
        self.epsilon = epsilon
        self.min_cluster_size = min_cluster_size
        self.logger = logging.getLogger(__name__)
    
    def plot_privacy_metrics(
        self,
        metrics: Dict[str, float],
        privacy_bounds: Dict[str, float]
    ) -> go.Figure:
        """Plot privacy metrics with bounds"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Privacy Metrics", "Privacy Bounds"]
        )
        
        # Privacy metrics
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                name="Metrics"
            ),
            row=1, col=1
        )
        
        # Privacy bounds
        fig.add_trace(
            go.Scatter(
                x=list(privacy_bounds.keys()),
                y=list(privacy_bounds.values()),
                mode='lines+markers',
                name="Bounds"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title="Privacy Analysis"
        )
        
        return fig
    
    def create_private_heatmap(
        self,
        data: torch.Tensor,
        title: str = "Privacy-Preserving Heatmap"
    ) -> go.Figure:
        """Generate privacy-preserving heatmap"""
        # Add noise and aggregate
        noisy_data = self._add_privacy_noise(data)
        aggregated_data = self._aggregate_sparse_regions(noisy_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=aggregated_data.numpy(),
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(title=title)
        return fig
    
    def _add_privacy_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add differentially private noise"""
        sensitivity = 1.0
        noise_scale = 2.0 * sensitivity / self.epsilon
        noise = torch.normal(0, noise_scale, size=data.shape)
        return data + noise
    
    def _aggregate_sparse_regions(self, data: torch.Tensor) -> torch.Tensor:
        """Aggregate sparse regions for privacy"""
        counts = torch.bincount(data.flatten().long())
        mask = counts >= self.min_cluster_size
        aggregated = data.clone()
        aggregated[~mask] = -1
        return aggregated

class CrossChainVisualizer:
    """Visualize cross-chain patterns and relationships"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def plot_cross_chain_patterns(
        self,
        chain_data: Dict[str, torch.Tensor],
        attention_weights: Dict[str, torch.Tensor]
    ) -> go.Figure:
        """Visualize cross-chain patterns"""
        num_chains = len(chain_data)
        chain_names = list(chain_data.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=num_chains,
            cols=num_chains,
            subplot_titles=[
                f"{chain1}-{chain2}"
                for chain1 in chain_names
                for chain2 in chain_names
            ]
        )
        
        # Add cross-chain visualizations
        for i, chain1 in enumerate(chain_names, 1):
            for j, chain2 in enumerate(chain_names, 1):
                attention_key = f"{chain1}-{chain2}"
                if attention_key in attention_weights:
                    self._add_attention_plot(
                        fig,
                        chain_data[chain1],
                        chain_data[chain2],
                        attention_weights[attention_key],
                        i, j
                    )
        
        fig.update_layout(
            height=300*num_chains,
            width=300*num_chains,
            title="Cross-Chain Pattern Analysis"
        )
        
        return fig
    
    def _add_attention_plot(
        self,
        fig: go.Figure,
        data1: torch.Tensor,
        data2: torch.Tensor,
        attention: torch.Tensor,
        row: int,
        col: int
    ) -> None:
        """Add attention visualization subplot"""
        fig.add_trace(
            go.Heatmap(
                z=attention.numpy(),
                colorscale='Viridis',
                showscale=True
            ),
            row=row,
            col=col
        )

class QuantumSecurityVisualizer:
    """Main quantum security visualization"""
    
    def __init__(self):
        self.quantum_metrics_plotter = QuantumMetricsPlotter()
        self.layer2_visualizer = Layer2Visualizer()
        self.privacy_visualizer = PrivacyVisualizer()
        self.cross_chain_visualizer = CrossChainVisualizer()
        self.logger = logging.getLogger(__name__)
    
    def visualize_quantum_security(
        self,
        metrics: Dict[str, float]
    ) -> go.Figure:
        """Generate comprehensive quantum security visualization"""
        return self.quantum_metrics_plotter.plot_quantum_security(metrics)
    
    def create_security_dashboard(
        self,
        quantum_metrics: Dict[str, float],
        layer2_metrics: Dict[str, float],
        privacy_metrics: Dict[str, float],
        cross_chain_data: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ) -> go.Figure:
        """Create comprehensive security visualization dashboard"""
        # Create dashboard with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Quantum Security Analysis",
                "Layer-2 Performance",
                "Privacy Metrics",
                "Cross-Chain Patterns"
            ]
        )
        
        # Add quantum security plot
        quantum_fig = self.quantum_metrics_plotter.plot_quantum_security(
            quantum_metrics
        )
        for trace in quantum_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add Layer-2 metrics
        layer2_fig = self.layer2_visualizer.plot_layer2_metrics(
            layer2_metrics
        )
        for trace in layer2_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Add privacy metrics
        privacy_fig = self.privacy_visualizer.plot_privacy_metrics(
            privacy_metrics,
            {'epsilon': CONFIG.model.privacy_epsilon}
        )
        for trace in privacy_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        # Add cross-chain patterns if available
        if cross_chain_data is not None:
            cross_chain_fig = self.cross_chain_visualizer.plot_cross_chain_patterns(
                cross_chain_data['data'],
                cross_chain_data['attention']
            )
            for trace in cross_chain_fig.data:
                fig.add_trace(trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1800,
            title="Comprehensive Security Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def generate_security_report(
        self,
        quantum_metrics: Dict[str, float],
        layer2_metrics: Dict[str, float],
        privacy_metrics: Dict[str, float]
    ) -> str:
        """Generate text report of security analysis"""
        report = []
        
        # Quantum security summary
        report.append("Quantum Security Analysis")
        report.append("=" * 30)
        for metric, value in quantum_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Layer-2 performance
        report.append("\nLayer-2 Performance")
        report.append("=" * 30)
        for metric, value in layer2_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Privacy analysis
        report.append("\nPrivacy Analysis")
        report.append("=" * 30)
        for metric, value in privacy_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        return "\n".join(report)

# Create factory function for visualizer
def create_visualizer(
    enable_privacy: bool = True,
    enable_layer2: bool = True,
    enable_cross_chain: bool = True
) -> QuantumSecurityVisualizer:
    """Create quantum security visualizer instance"""
    return QuantumSecurityVisualizer()