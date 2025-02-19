import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from torch_geometric.data import Data, Batch
from config import CONFIG
from hierarchical_learn import (
    QuantumResistantHierarchicalLearning,
    Layer2HierarchicalLearning
)
from privacy_layer import PrivacyLayer

class QuantumResistantLayers(nn.Module):
    """Quantum-resistant neural network layers"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Lattice-based parameters
        self.lattice_dim = CONFIG.quantum.lattice_dimension
        self.noise_scale = CONFIG.quantum.post_quantum_params["noise_width"]
        
        # Quantum-resistant layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                QuantumResistantLinear(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    self.lattice_dim
                ),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum resistance"""
        for layer in self.layers:
            # Add quantum noise
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
            
            # Apply transformation
            x = layer(x)
        return x

class QuantumResistantLinear(nn.Module):
    """Quantum-resistant linear transformation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lattice_dim: int
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lattice_dim = lattice_dim
        
        # Initialize lattice-based weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Noise parameter
        self.register_buffer(
            'noise_scale',
            torch.tensor(math.sqrt(lattice_dim) / out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with lattice-based transformation"""
        # Add weight noise
        weight_noise = torch.randn_like(self.weight) * self.noise_scale
        noisy_weight = self.weight + weight_noise
        
        return F.linear(x, noisy_weight, self.bias)

class Layer2Processor(nn.Module):
    """Process Layer-2 data with rollups"""
    
    def __init__(
        self,
        hidden_dim: int,
        compression_ratio: float = CONFIG.model.layer2_config.compression_ratio
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compression_ratio = compression_ratio
        
        # Compression network
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, int(hidden_dim / compression_ratio))
        )
        
        # Decompression network
        self.decompress = nn.Sequential(
            nn.Linear(int(hidden_dim / compression_ratio), hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process data through compression/decompression"""
        compressed = self.compress(x)
        decompressed = self.decompress(compressed)
        return compressed, decompressed

class EnhancedCrossChainAttention(nn.Module):
    """Enhanced cross-chain attention mechanism"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_chains: int = len(CONFIG.data.chains),
        num_heads: int = CONFIG.model.cross_chain_attention_heads,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_chains = num_chains
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        
        # Chain embeddings
        self.chain_embeddings = nn.Parameter(
            torch.randn(num_chains, hidden_dim)
        )
        
        # Quantum-resistant attention
        self.attention = QuantumResistantAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Chain mixing
        self.chain_mixing = nn.Parameter(
            torch.randn(num_chains, num_chains)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        layer2_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-chain attention"""
        chain_features = []
        chain_ids = list(features.keys())
        
        # Add chain embeddings
        for i, (chain, feat) in enumerate(features.items()):
            chain_feat = feat + self.chain_embeddings[i]
            chain_features.append(chain_feat)
        
        # Process Layer-2 features if available
        if layer2_features is not None:
            for chain, l2_feat in layer2_features.items():
                if chain in features:
                    idx = chain_ids.index(chain)
                    chain_features[idx] = chain_features[idx] + l2_feat
        
        # Apply cross-chain attention
        mixed_features = {}
        for i, chain in enumerate(chain_ids):
            # Compute attention
            attended = self.attention(
                chain_features[i],
                torch.stack(chain_features),
                torch.stack(chain_features)
            )
            
            # Apply chain mixing
            mixed = torch.sum(
                attended * self.chain_mixing[i].view(-1, 1, 1),
                dim=0
            )
            
            # Residual connection and normalization
            mixed_features[chain] = self.layer_norm(
                features[chain] + self.dropout(mixed)
            )
        
        return mixed_features

class QuantumResistantHIPADual(nn.Module):
    """Enhanced HIPADual with quantum resistance"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        num_classes: int = 2,
        layer2_enabled: bool = True
    ):
        super().__init__()
        
        # Initialize components
        self.quantum_layers = QuantumResistantLayers(
            input_dim=in_channels,
            hidden_dim=hidden_channels[-1]
        )
        
        self.hierarchical_learner = QuantumResistantHierarchicalLearning(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            layer2_enabled=layer2_enabled
        )
        
        if layer2_enabled:
            self.layer2_processor = Layer2Processor(
                hidden_dim=hidden_channels[-1]
            )
        
        self.cross_chain_attention = EnhancedCrossChainAttention(
            hidden_dim=hidden_channels[-1]
        )
        
        self.privacy_layer = PrivacyLayer()
        
        # Chain-specific processors
        self.chain_processors = nn.ModuleDict({
            chain: nn.Sequential(
                QuantumResistantLinear(
                    hidden_channels[-1],
                    hidden_channels[-1],
                    CONFIG.quantum.lattice_dimension
                ),
                nn.LayerNorm(hidden_channels[-1]),
                nn.GELU(),
                nn.Dropout(CONFIG.model.dropout)
            )
            for chain in CONFIG.data.chains
        })
        
        # Output layers
        self.classifier = nn.Sequential(
            QuantumResistantLinear(
                hidden_channels[-1],
                hidden_channels[-1] // 2,
                CONFIG.quantum.lattice_dimension
            ),
            nn.LayerNorm(hidden_channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(CONFIG.model.dropout),
            nn.Linear(hidden_channels[-1] // 2, num_classes)
        )
        
        # Initialize confidence estimation
        self.confidence_estimator = nn.Sequential(
            QuantumResistantLinear(
                hidden_channels[-1],
                hidden_channels[-1] // 4,
                CONFIG.quantum.lattice_dimension
            ),
            nn.LayerNorm(hidden_channels[-1] // 4),
            nn.GELU(),
            nn.Dropout(CONFIG.model.dropout),
            nn.Linear(hidden_channels[-1] // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        data: Union[Dict[str, Data], Dict[str, Batch]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with quantum resistance"""
        # Dictionary to store results per chain
        chain_results = {}
        chain_features = {}
        layer2_features = {}
        
        # Process each chain
        for chain_name, chain_data in data.items():
            # Apply privacy preservation
            x_private, edge_index, edge_attr = self.privacy_layer(
                chain_data.x,
                chain_data.edge_index,
                chain_data.edge_attr if hasattr(chain_data, 'edge_attr') else None
            )
            
            # Apply quantum-resistant feature extraction
            x_quantum = self.quantum_layers(x_private)
            
            # Process Layer-2 if available
            if hasattr(chain_data, 'layer2_data'):
                l2_x, l2_edge_index = self.layer2_processor(x_quantum)
                hierarchical_output, l2_output = self.hierarchical_learner(
                    x_quantum,
                    edge_index,
                    (l2_x, l2_edge_index),
                    chain_data.batch if hasattr(chain_data, 'batch') else None
                )
                layer2_features[chain_name] = l2_output
            else:
                hierarchical_output = self.hierarchical_learner(
                    x_quantum,
                    edge_index,
                    None,
                    chain_data.batch if hasattr(chain_data, 'batch') else None
                )
            
            # Apply chain-specific processing
            chain_features[chain_name] = self.chain_processors[chain_name](
                hierarchical_output
            )
        
        # Apply cross-chain attention
        mixed_features = self.cross_chain_attention(
            chain_features,
            layer2_features if layer2_features else None
        )
        
        # Generate predictions for each chain
        for chain_name, features in mixed_features.items():
            logits = self.classifier(features)
            confidence = self.confidence_estimator(features)
            chain_results[chain_name] = (logits, confidence)
        
        return chain_results
    
    def get_interpretability_info(
        self,
        data: Dict[str, Data]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get interpretability information for predictions"""
        interpretability_info = {}
        
        # Process each chain
        for chain_name, chain_data in data.items():
            chain_info = {}
            
            # Get hierarchical attention weights
            if hasattr(chain_data, 'layer2_data'):
                l2_x, l2_edge_index = self.layer2_processor(chain_data.x)
                _, hierarchical_features = self.hierarchical_learner(
                    chain_data.x,
                    chain_data.edge_index,
                    (l2_x, l2_edge_index)
                )
            else:
                _, hierarchical_features = self.hierarchical_learner(
                    chain_data.x,
                    chain_data.edge_index,
                    None
                )
            
            # Store interpretability information
            chain_info['hierarchical_features'] = hierarchical_features
            chain_info['attention_weights'] = self.cross_chain_attention.chain_mixing[
                list(data.keys()).index(chain_name)
            ]
            
            interpretability_info[chain_name] = chain_info
        
        return interpretability_info
    
    def get_quantum_security_metrics(self) -> Dict[str, float]:
        """Get quantum security-related metrics"""
        security_metrics = {
            'lattice_dimension': CONFIG.quantum.lattice_dimension,
            'noise_scale': CONFIG.quantum.post_quantum_params["noise_width"],
            'quantum_security_level': self._estimate_security_level()
        }
        
        return security_metrics
    
    def _estimate_security_level(self) -> float:
        """Estimate quantum security level"""
        # Consider lattice parameters and noise levels
        lattice_security = math.log2(CONFIG.quantum.lattice_dimension)
        noise_security = -math.log2(CONFIG.quantum.post_quantum_params["noise_width"])
        
        return min(lattice_security, noise_security)