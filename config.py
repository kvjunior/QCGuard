import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import torch

@dataclass
class QuantumConfig:
    """Quantum resistance configurations"""
    security_level: int = 256
    lattice_dimension: int = 1024
    quantum_resistant_scheme: str = "kyber"  # or "dilithium"
    noise_distribution: str = "gaussian"
    key_generation_method: str = "lattice"
    post_quantum_params: Dict[str, Any] = field(default_factory=lambda: {
        "modulus": 2**32 - 5,
        "noise_width": 3.2,
        "ring_dimension": 1024,
        "error_distribution": "discrete_gaussian"
    })
    quantum_safe_hash: str = "SHAKE256"
    quantum_rng_seed: int = 42

@dataclass
class Layer2Config:
    """Layer-2 and rollup configurations"""
    enabled: bool = True
    rollup_type: str = "zk"  # or "optimistic"
    batch_size: int = 10000
    proof_generation_method: str = "snark"  # or "stark"
    zk_proof_params: Dict[str, Any] = field(default_factory=lambda: {
        "curve": "bn254",
        "constraints": 2**20,
        "security_level": 128
    })
    optimistic_params: Dict[str, Any] = field(default_factory=lambda: {
        "challenge_period": 7 * 24 * 3600,  # 1 week in seconds
        "bond_amount": 1000  # in base units
    })
    compression_ratio: float = 10.0
    max_batch_gas: int = 10000000

@dataclass
class EnhancedPrivacyConfig:
    """Enhanced privacy configurations for 2025"""
    epsilon: float = 0.1
    delta: float = 1e-8
    noise_multiplier: float = 1.5
    clipping_threshold: float = 1.0
    min_batch_size: int = 1000
    privacy_accountant: str = "rdp"  # or "moments", "gdp"
    secure_aggregation_threshold: int = 100
    homomorphic_encryption_params: Dict[str, Any] = field(default_factory=lambda: {
        "scheme": "CKKS",
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60]
    })
    mpc_protocol: str = "shamir"  # or "bmr", "yao"
    zero_knowledge_params: Dict[str, Any] = field(default_factory=lambda: {
        "proving_system": "groth16",
        "curve": "bn254",
        "security_bits": 128
    })

@dataclass
class ScalabilityConfig:
    """Enhanced scalability configurations"""
    max_nodes: int = 10000
    shard_count: int = 32
    cross_shard_tx_ratio: float = 0.1
    max_concurrent_users: int = 1000000
    throughput_tps: int = 100000
    latency_ms: int = 100
    storage_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "compression": "snappy",
        "pruning_depth": 1000,
        "state_cache_size": 10000
    })

@dataclass
class DataConfig:
    """Data and preprocessing configurations"""
    data_dir: str = "./data"
    supported_chains: List[str] = field(default_factory=lambda: [
        "ethereum", "bitcoin", "polygon", "arbitrum", "optimism"
    ])
    chain_specific_paths: Dict[str, str] = field(default_factory=lambda: {
        chain: os.path.join("./data", f"{chain}.pt") for chain in [
            "ethereum_s", "ethereum_p", "bitcoin_m", "bitcoin_l"
        ]
    })
    bridge_contracts: Dict[str, List[str]] = field(default_factory=lambda: {
        "ethereum": ["0x123...", "0x456..."],
        "bitcoin": ["bc1...", "3..."],
        "polygon": ["0x789...", "0xabc..."]
    })
    sequence_settings: Dict[str, int] = field(default_factory=lambda: {
        "max_length": 128,
        "min_length": 32,
        "padding_value": 0
    })
    feature_dimensions: Dict[str, int] = field(default_factory=lambda: {
        "node": 128,
        "edge": 64,
        "temporal": 32
    })
    sampling_params: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "num_neighbors": 25,
        "neg_pos_ratio": 3.0,
        "max_positive_samples": 10000
    })

@dataclass
class ModelConfig:
    """Enhanced model architecture configurations"""
    quantum_resistant: bool = True
    quantum_config: QuantumConfig = field(default_factory=QuantumConfig)
    layer2_config: Layer2Config = field(default_factory=Layer2Config)
    
    # Architecture params
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 16
    dropout_rate: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    
    # Attention mechanisms
    attention_types: List[str] = field(default_factory=lambda: [
        "self", "cross", "global"
    ])
    attention_params: Dict[str, Any] = field(default_factory=lambda: {
        "head_dim": 64,
        "dropout": 0.1,
        "scaled": True,
        "causal": False
    })
    
    # Cross-chain analysis
    cross_chain_heads: int = 8
    chain_embedding_dim: int = 64
    chain_specific_layers: int = 3

@dataclass
class TrainingConfig:
    """Enhanced training configurations"""
    # Basic training params
    batch_size: int = 1024
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Advanced training features
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_gradient_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.01
    
    # Distributed training
    distributed_backend: str = "nccl"
    num_workers: int = 8
    local_rank: int = -1
    find_unused_parameters: bool = False
    
    # Privacy and security
    privacy_config: EnhancedPrivacyConfig = field(default_factory=EnhancedPrivacyConfig)
    secure_aggregation: bool = True
    quantum_safe_training: bool = True

@dataclass
class EvaluationConfig:
    """Enhanced evaluation configurations"""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1",
        "auc", "average_precision", "quantum_security",
        "privacy_guarantee", "cross_chain_consistency"
    ])
    eval_frequency: int = 100
    eval_samples: int = 10000
    quantum_eval_rounds: int = 10
    privacy_test_cases: List[str] = field(default_factory=lambda: [
        "membership_inference", "model_inversion",
        "attribute_inference", "quantum_attack"
    ])
    cross_chain_metrics: List[str] = field(default_factory=lambda: [
        "pattern_consistency", "temporal_correlation",
        "bridge_detection", "anomaly_detection"
    ])

@dataclass
class Config:
    """Master configuration class combining all sub-configs"""
    # Device configuration
    device: torch.device = field(default_factory=lambda: 
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    precision: str = "float32"
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    scalability: ScalabilityConfig = field(default_factory=ScalabilityConfig)
    
    # System paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        """Perform post-initialization setup"""
        # Create necessary directories
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set random seeds for reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with custom parameters"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def save(self, path: str) -> None:
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        config.update(config_dict)
        return config

# Create global configuration instance
CONFIG = Config()