# QCGuard: Quantum-Resistant Cross-Chain Security Framework

QCGuard is a quantum-resistant framework for securing cross-chain transactions through Byzantine fault-tolerant federated learning. It achieves 0.981-0.991 Grover attack resistance, differential privacy guarantees (ε ≤ 0.098, δ ≤ 2.47e-8), and 0.963-0.984 cross-chain consistency.

## Key Features

- **Quantum Resistance**: Lattice-based security layer providing robust protection against quantum attacks
- **Privacy Preservation**: Advanced differential privacy mechanisms with strong guarantees
- **Cross-Chain Security**: Byzantine fault-tolerant federated learning for secure cross-chain transactions
- **Layer-2 Scalability**: Efficient compression (10.7x-12.4x) with zero-knowledge proofs
- **High Performance**: 3,840-4,820 transactions/second with 0.927 linear scaling efficiency

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA toolkit 12.0 or higher (for GPU support)
- 16GB RAM (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM

### Setup

1. Clone the repository:


2. Create and activate virtual environment:
python -m venv env
source env/bin/activate  # Linux/Mac
# OR
env\Scripts\activate     # Windows


3. Install dependencies:
pip install -r requirements.txt


## Usage

### Basic Training

from trainer import QuantumResistantTrainer
from model import QuantumResistantHIPADual
from config import CONFIG

# Initialize model
model = QuantumResistantHIPADual(
    in_channels=CONFIG.data.node_feature_dim,
    hidden_channels=CONFIG.model.level_dims
)

# Create trainer
trainer = QuantumResistantTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)

# Train model
trainer.train(num_epochs=CONFIG.training.num_epochs)


### Evaluating Security

from evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(
    model=model,
    test_data=test_data,
    layer2_data=layer2_data,
    device=device
)


## Dataset Support

The framework has been evaluated on four major cryptocurrency networks:
- Ethereum-S (1.3M nodes)
- Ethereum-P (3.0M nodes)
- Bitcoin-M (2.5M nodes)
- Bitcoin-L (20.1M nodes)

## Performance Results

### Security Metrics
- Quantum Resistance: 0.981-0.991 Grover attack resistance
- Privacy Guarantees: ε ≤ 0.098, δ ≤ 2.47e-8
- Cross-Chain Consistency: 0.963-0.984

### Processing Performance
- Throughput: 3,840-4,820 tx/s
- Layer-2 Compression: 10.7x-12.4x
- Linear Scaling Efficiency: 0.927

## Configuration

Configuration can be modified in `config.py`. Key parameters include:
- Quantum security level
- Privacy parameters (ε, δ)
- Layer-2 compression ratio
- Training hyperparameters

## Visualization

from visualizer import QuantumSecurityVisualizer

visualizer = QuantumSecurityVisualizer()
dashboard = visualizer.create_security_dashboard(
    quantum_metrics=results['quantum'],
    layer2_metrics=results['layer2'],
    privacy_metrics=results['privacy']
)


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request