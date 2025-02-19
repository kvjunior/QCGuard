import torch
import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import wandb
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from config import CONFIG, update_config
from model import QuantumResistantHIPADual
from data_processor import EnhancedBlockchainDataProcessor
from trainer import QuantumResistantTrainer
from evaluator import Evaluator
from visualizer import QuantumSecurityVisualizer
from security_proofs import SECURITY_PROOFS

class QuantumInitializer:
    """Initialize quantum-resistant components"""
    
    def __init__(self, security_level: int = CONFIG.quantum.security_level):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
    def initialize_quantum_components(
        self,
        model: torch.nn.Module
    ) -> torch.nn.Module:
        """Initialize quantum-resistant model components"""
        # Set quantum-resistant parameters
        self._set_quantum_parameters(model)
        
        # Initialize quantum-resistant layers
        self._initialize_quantum_layers(model)
        
        # Verify initialization
        if not self._verify_quantum_initialization(model):
            raise ValueError("Quantum initialization failed")
        
        return model
    
    def _set_quantum_parameters(self, model: torch.nn.Module) -> None:
        """Set quantum-resistant parameters"""
        for param in model.parameters():
            with torch.no_grad():
                # Apply lattice-based initialization
                noise = torch.randn_like(param) * \
                        CONFIG.quantum.post_quantum_params["noise_width"]
                param.add_(noise)
    
    def _initialize_quantum_layers(self, model: torch.nn.Module) -> None:
        """Initialize quantum-resistant layers"""
        for module in model.modules():
            if hasattr(module, 'quantum_resistant'):
                module.quantum_resistant = True
    
    def _verify_quantum_initialization(self, model: torch.nn.Module) -> bool:
        """Verify quantum-resistant initialization"""
        try:
            # Verify parameters
            for param in model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Quantum verification failed: {str(e)}")
            return False

class Layer2Manager:
    """Manage Layer-2 operations and rollups"""
    
    def __init__(self):
        self.rollup_batch_size = CONFIG.model.layer2_config.batch_size
        self.compression_ratio = CONFIG.model.layer2_config.compression_ratio
        self.logger = logging.getLogger(__name__)
        
    def process_layer2_data(
        self,
        data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process data through Layer-2"""
        metrics = {}
        
        # Compress data for rollup
        compressed_data = self._compress_data(data)
        metrics['compression_ratio'] = self._compute_compression_ratio(
            data,
            compressed_data
        )
        
        # Generate rollup proof
        proof = self._generate_rollup_proof(data, compressed_data)
        metrics['proof_size'] = len(str(proof))
        
        return compressed_data, metrics
    
    def _compress_data(self, data: torch.Tensor) -> torch.Tensor:
        """Compress data for rollup"""
        return F.adaptive_avg_pool1d(
            data.unsqueeze(0),
            int(data.size(-1) / self.compression_ratio)
        ).squeeze(0)
    
    def _generate_rollup_proof(
        self,
        original_data: torch.Tensor,
        compressed_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate zero-knowledge rollup proof"""
        return {}  # Placeholder for actual proof generation
    
    def _compute_compression_ratio(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor
    ) -> float:
        """Compute actual compression ratio"""
        return original.numel() / compressed.numel()

class ExperimentTracker:
    """Enhanced experiment tracking and monitoring"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = {}
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self) -> None:
        """Setup experiment monitoring"""
        if CONFIG.training.use_wandb:
            wandb.init(
                project="hipadual",
                name=self.experiment_name,
                config=CONFIG
            )
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log experiment metrics"""
        # Update internal metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to wandb if enabled
        if CONFIG.training.use_wandb:
            wandb.log(metrics)
        
        # Log to console
        self.logger.info(
            "Metrics: " + " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
    
    def save_experiment_results(self, path: Path) -> None:
        """Save experiment results"""
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': self.metrics,
            'config': CONFIG.__dict__
        }
        
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

class EnhancedExperimentRunner:
    """Enhanced experiment runner with quantum support"""
    
    def __init__(self, experiment_name: str):
        self.quantum_initializer = QuantumInitializer()
        self.layer2_manager = Layer2Manager()
        self.experiment_tracker = ExperimentTracker(experiment_name)
        self.visualizer = QuantumSecurityVisualizer()
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = Path("experiment_results") / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_enhanced_evaluation(
        self,
        model: torch.nn.Module,
        data: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, Any]:
        """Run enhanced evaluation suite"""
        results = {}
        
        try:
            # Initialize quantum components
            model = self.quantum_initializer.initialize_quantum_components(model)
            
            # Process Layer-2 data if enabled
            if CONFIG.model.layer2_config.enabled:
                l2_data = {}
                l2_metrics = {}
                for chain, chain_data in data.items():
                    l2_data[chain], metrics = self.layer2_manager.process_layer2_data(
                        chain_data
                    )
                    l2_metrics[chain] = metrics
                results['layer2'] = l2_metrics
            
            # Run evaluation
            evaluator = Evaluator()
            eval_results = evaluator.evaluate(
                model,
                data,
                l2_data if CONFIG.model.layer2_config.enabled else None,
                device
            )
            results['evaluation'] = eval_results
            
            # Create visualizations
            self._generate_visualizations(
                eval_results,
                l2_metrics if CONFIG.model.layer2_config.enabled else None
            )
            
            # Track results
            self.experiment_tracker.log_metrics(eval_results)
            
            # Save results
            self.experiment_tracker.save_experiment_results(self.output_dir)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _generate_visualizations(
        self,
        eval_results: Dict[str, float],
        l2_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """Generate experiment visualizations"""
        # Create security dashboard
        dashboard = self.visualizer.create_security_dashboard(
            eval_results.get('quantum', {}),
            l2_metrics if l2_metrics else {},
            eval_results.get('privacy', {}),
            eval_results.get('cross_chain', {})
        )
        
        # Save visualization
        dashboard.write_html(str(self.output_dir / "security_dashboard.html"))
        
        # Generate and save security report
        report = self.visualizer.generate_security_report(
            eval_results.get('quantum', {}),
            l2_metrics if l2_metrics else {},
            eval_results.get('privacy', {})
        )
        
        with open(self.output_dir / "security_report.txt", 'w') as f:
            f.write(report)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HIPADual Enhanced Experiment Runner')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name for the experiment')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    # Update configuration if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_updates = json.load(f)
        update_config(config_updates)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Initialize data processor
    data_processor = EnhancedBlockchainDataProcessor("layer2_enabled")
    
    # Load and process data
    data = data_processor.process_data({
        chain: torch.load(path)
        for chain, path in CONFIG.data.chain_specific_paths.items()
    })
    
    # Initialize model
    model = QuantumResistantHIPADual(
        in_channels=CONFIG.data.node_feature_dim,
        hidden_channels=CONFIG.model.level_dims
    ).to(device)
    
    # Create experiment runner
    runner = EnhancedExperimentRunner(args.experiment_name)
    
    if not args.eval_only:
        # Initialize trainer
        trainer = QuantumResistantTrainer(
            model=model,
            train_loader=DataLoader(data['train']),
            val_loader=DataLoader(data['val']),
            device=device
        )
        
        # Train model
        trainer.train(
            num_epochs=CONFIG.training.num_epochs,
            save_path=str(runner.output_dir / "model_best.pt")
        )
    
    # Run evaluation
    results = runner.run_enhanced_evaluation(model, data['test'], device)
    
    # Save final results
    with open(runner.output_dir / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()