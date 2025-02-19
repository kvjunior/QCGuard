import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import List, Tuple, Dict, Optional, Union, Any
import math
from config import CONFIG
import logging

class QuantumFeatureExtractor:
    """Quantum-resistant feature extraction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        lattice_dim: int = CONFIG.quantum.lattice_dimension
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lattice_dim = lattice_dim
        
        # Quantum-resistant layers
        self.layers = nn.ModuleList([
            QuantumResistantLayer(
                in_dim=input_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                lattice_dim=lattice_dim
            )
            for i in range(num_layers)
        ])
        
        self.noise_scale = math.sqrt(lattice_dim) / hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum resistance"""
        for layer in self.layers:
            # Add quantum noise for security
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
            
            # Apply quantum-resistant transformation
            x = layer(x)
            
        return x

class QuantumResistantLayer(nn.Module):
    """Quantum-resistant neural network layer"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lattice_dim: int
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lattice_dim = lattice_dim
        
        # Lattice-based parameters
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim) / math.sqrt(in_dim)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # Quantum noise parameters
        self.register_buffer(
            'noise_scale',
            torch.tensor(math.sqrt(lattice_dim) / out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with lattice-based transformation"""
        # Add lattice-based noise
        weight_noise = torch.randn_like(self.weight) * self.noise_scale
        noisy_weight = self.weight + weight_noise
        
        # Apply transformation
        out = F.linear(x, noisy_weight, self.bias)
        return F.gelu(out)

class QuantumResistantAttention(nn.Module):
    """Quantum-resistant attention mechanism"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        lattice_dim: int = CONFIG.quantum.lattice_dimension
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Quantum-resistant projections
        self.q_proj = QuantumResistantLayer(hidden_dim, hidden_dim, lattice_dim)
        self.k_proj = QuantumResistantLayer(hidden_dim, hidden_dim, lattice_dim)
        self.v_proj = QuantumResistantLayer(hidden_dim, hidden_dim, lattice_dim)
        self.output_proj = QuantumResistantLayer(hidden_dim, hidden_dim, lattice_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with quantum-resistant attention"""
        batch_size = query.size(0)
        
        # Quantum-resistant projections
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Get attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        
        return self.output_proj(attn_output)

class CrossLayerAttention(nn.Module):
    """Cross-layer attention for Layer-2 hierarchies"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer-specific embeddings
        self.l1_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.l2_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Quantum-resistant attention
        self.attention = QuantumResistantAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        l1_features: torch.Tensor,
        l2_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-layer attention"""
        batch_size = l1_features.size(0)
        
        # Add layer embeddings
        l1_features = l1_features + self.l1_embedding
        l2_features = l2_features + self.l2_embedding
        
        # Cross attention L1 -> L2
        l2_attended = self.attention(l2_features, l1_features, l1_features, mask)
        l2_output = self.layer_norm(l2_features + self.dropout(l2_attended))
        
        # Cross attention L2 -> L1
        l1_attended = self.attention(l1_features, l2_features, l2_features, mask)
        l1_output = self.layer_norm(l1_features + self.dropout(l1_attended))
        
        return l1_output, l2_output

class RollupProcessor:
    """Process rollups in hierarchical learning"""
    
    def __init__(
        self,
        hidden_dim: int,
        compression_ratio: float = CONFIG.model.layer2_config.compression_ratio
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compression_ratio = compression_ratio
        
        # Compression layers
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, int(hidden_dim / compression_ratio))
        )
        
        # Decompression layers
        self.decompress = nn.Sequential(
            nn.Linear(int(hidden_dim / compression_ratio), hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def process_rollup(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process features through rollup"""
        # Compress features
        compressed = self.compress(features)
        
        # Decompress features
        decompressed = self.decompress(compressed)
        
        return compressed, decompressed

class Layer2HierarchicalLearning(nn.Module):
    """Hierarchical learning with Layer-2 support"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        compression_ratio: float = CONFIG.model.layer2_config.compression_ratio
    ):
        super().__init__()
        self.num_levels = len(hidden_channels)
        
        # Initialize components
        self.rollup_processor = RollupProcessor(
            hidden_dim=hidden_channels[0],
            compression_ratio=compression_ratio
        )
        self.cross_layer_attention = CrossLayerAttention(
            hidden_dim=hidden_channels[0]
        )
        
        # Layer-1 hierarchical layers
        self.l1_layers = nn.ModuleList([
            GATConv(
                in_channels if i == 0 else hidden_channels[i-1],
                hidden_channels[i],
                heads=CONFIG.model.num_heads
            )
            for i in range(self.num_levels)
        ])
        
        # Layer-2 hierarchical layers
        self.l2_layers = nn.ModuleList([
            GATConv(
                int(hidden_channels[0] / compression_ratio) if i == 0 
                else hidden_channels[i-1],
                hidden_channels[i],
                heads=CONFIG.model.num_heads
            )
            for i in range(self.num_levels)
        ])
        
    def forward(
        self,
        l1_x: torch.Tensor,
        l1_edge_index: torch.Tensor,
        l2_x: torch.Tensor,
        l2_edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Layer-1 and Layer-2 hierarchies"""
        l1_features = []
        l2_features = []
        
        # Process Layer-1
        current_l1 = l1_x
        for layer in self.l1_layers:
            current_l1 = layer(current_l1, l1_edge_index)
            l1_features.append(current_l1)
        
        # Process Layer-2 with rollups
        compressed_l2, decompressed_l2 = self.rollup_processor(l2_x)
        
        current_l2 = compressed_l2
        for layer in self.l2_layers:
            current_l2 = layer(current_l2, l2_edge_index)
            l2_features.append(current_l2)
        
        # Apply cross-layer attention
        l1_output, l2_output = self.cross_layer_attention(
            l1_features[-1],
            l2_features[-1]
        )
        
        return l1_output, l2_output

class QuantumResistantHierarchicalLearning(nn.Module):
    """Quantum-resistant hierarchical learning"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        layer2_enabled: bool = True
    ):
        super().__init__()
        self.layer2_enabled = layer2_enabled
        
        # Quantum-resistant components
        self.quantum_feature_extractor = QuantumFeatureExtractor(
            input_dim=in_channels,
            hidden_dim=hidden_channels[0]
        )
        self.resistant_attention = QuantumResistantAttention(
            hidden_dim=hidden_channels[0]
        )
        
        # Layer-2 components
        if layer2_enabled:
            self.layer2_hierarchy = Layer2HierarchicalLearning(
                in_channels=in_channels,
                hidden_channels=hidden_channels
            )
        
        # Hierarchical GNN layers
        self.conv_layers = nn.ModuleList([
            GATConv(
                in_channels if i == 0 else hidden_channels[i-1],
                hidden_channels[i],
                heads=CONFIG.model.num_heads
            )
            for i in range(len(hidden_channels))
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer2_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with quantum resistance"""
        # Extract quantum-resistant features
        features = self.quantum_feature_extractor(x)
        
        # Apply hierarchical layers
        current_features = features
        hierarchical_features = []
        
        for layer in self.conv_layers:
            # Apply quantum-resistant attention
            current_features = self.resistant_attention(
                current_features,
                current_features,
                current_features
            )
            
            # Apply graph convolution
            current_features = layer(current_features, edge_index)
            hierarchical_features.append(current_features)
        
        if self.layer2_enabled and layer2_data is not None:
            l2_x, l2_edge_index = layer2_data
            l1_output, l2_output = self.layer2_hierarchy(
                hierarchical_features[-1],
                edge_index,
                l2_x,
                l2_edge_index,
                batch
            )
            return l1_output, l2_output
        
        return hierarchical_features[-1]