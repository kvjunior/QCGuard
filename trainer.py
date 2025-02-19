import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from config import CONFIG
import time
from datetime import datetime
from privacy_layer import PrivacyLayer
from security_proofs import SECURITY_PROOFS

class QuantumResistantOptimizer:
    """Quantum-resistant optimization wrapper"""
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        noise_scale: float = CONFIG.quantum.post_quantum_params["noise_width"],
        max_grad_norm: float = 1.0
    ):
        self.optimizer = base_optimizer
        self.noise_scale = noise_scale
        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler()
        self.steps = 0
        self.logger = logging.getLogger(__name__)
    
    def step(self, loss: torch.Tensor) -> None:
        """Perform quantum-resistant optimization step"""
        # Scale loss for mixed precision
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Unscale gradients
        self.scaler.unscale_(self.optimizer)
        
        # Add quantum noise to gradients
        self._add_quantum_noise()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'],
            self.max_grad_norm
        )
        
        # Step optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.steps += 1
    
    def _add_quantum_noise(self) -> None:
        """Add quantum-resistant noise to gradients"""
        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None:
                # Generate lattice-based noise
                noise = torch.randn_like(param.grad) * self.noise_scale
                param.grad += noise

class Layer2Trainer:
    """Layer-2 specific training implementation"""
    
    def __init__(
        self,
        compression_ratio: float = CONFIG.model.layer2_config.compression_ratio,
        batch_size: int = CONFIG.model.layer2_config.batch_size
    ):
        self.compression_ratio = compression_ratio
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def train_layer2(
        self,
        model: nn.Module,
        l1_data: torch.Tensor,
        l2_data: torch.Tensor,
        optimizer: QuantumResistantOptimizer
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train model on Layer-2 data"""
        model.train()
        metrics = {}
        
        # Process L2 data in batches
        l2_batches = torch.split(l2_data, self.batch_size)
        l1_batches = torch.split(l1_data, self.batch_size)
        
        total_loss = 0
        for l1_batch, l2_batch in zip(l1_batches, l2_batches):
            # Forward pass through both layers
            l1_out, l2_out = model(l1_batch, l2_batch)
            
            # Compute Layer-2 specific loss
            loss = self._compute_l2_loss(l1_out, l2_out, l1_batch, l2_batch)
            
            # Update model
            optimizer.step(loss)
            
            total_loss += loss.item()
            
        # Compute metrics
        metrics['l2_loss'] = total_loss / len(l2_batches)
        metrics['compression_rate'] = self._compute_compression_rate(l2_data)
        
        return model, metrics
    
    def _compute_l2_loss(
        self,
        l1_out: torch.Tensor,
        l2_out: torch.Tensor,
        l1_data: torch.Tensor,
        l2_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute Layer-2 specific loss"""
        # Reconstruction loss
        recon_loss = F.mse_loss(l2_out, l2_data)
        
        # Cross-layer consistency loss
        consistency_loss = F.mse_loss(l1_out, l2_out)
        
        # Compression loss
        compression_loss = torch.norm(l2_out) * self.compression_ratio
        
        return recon_loss + 0.1 * consistency_loss + 0.01 * compression_loss
    
    def _compute_compression_rate(self, l2_data: torch.Tensor) -> float:
        """Compute actual compression rate"""
        return l2_data.numel() / (l2_data.numel() * self.compression_ratio)

class PrivacyPreservingTrainer:
    """Enhanced privacy-preserving training"""
    
    def __init__(self):
        self.privacy_layer = PrivacyLayer()
        self.epsilon_budget = CONFIG.model.privacy_epsilon
        self.accumulated_cost = 0.0
        self.logger = logging.getLogger(__name__)
    
    def train_with_privacy(
        self,
        model: nn.Module,
        data: torch.Tensor,
        optimizer: QuantumResistantOptimizer
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train model with privacy preservation"""
        model.train()
        metrics = {}
        
        # Apply differential privacy
        private_data = self.privacy_layer(data)
        
        # Forward pass
        outputs = model(private_data)
        
        # Compute private loss
        loss = self._compute_private_loss(outputs, data)
        
        # Update model
        optimizer.step(loss)
        
        # Track privacy cost
        privacy_cost = self._compute_privacy_cost(outputs)
        self.accumulated_cost += privacy_cost
        
        # Update metrics
        metrics['privacy_loss'] = loss.item()
        metrics['privacy_cost'] = privacy_cost
        metrics['remaining_budget'] = self.epsilon_budget - self.accumulated_cost
        
        return model, metrics
    
    def _compute_private_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute privacy-aware loss"""
        # Standard loss
        base_loss = F.cross_entropy(outputs, targets)
        
        # Add privacy regularization
        privacy_reg = self.privacy_layer.get_privacy_regularization(outputs)
        
        return base_loss + 0.1 * privacy_reg
    
    def _compute_privacy_cost(self, outputs: torch.Tensor) -> float:
        """Compute privacy cost of current step"""
        return self.privacy_layer.compute_privacy_cost(outputs)

class QuantumResistantTrainer:
    """Main quantum-resistant training implementation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        distributed: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.distributed = distributed
        
        # Setup components
        self.quantum_optimizer = QuantumResistantOptimizer(
            optim.AdamW(model.parameters())
        )
        self.layer2_trainer = Layer2Trainer()
        self.privacy_trainer = PrivacyPreservingTrainer()
        
        # Setup distributed training
        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model)
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics = {}
        
        for batch in tqdm(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Layer-2 training if available
            if 'l2_data' in batch:
                self.model, l2_metrics = self.layer2_trainer.train_layer2(
                    self.model,
                    batch['data'],
                    batch['l2_data'],
                    self.quantum_optimizer
                )
                metrics.update(l2_metrics)
            
            # Privacy-preserving training
            self.model, privacy_metrics = self.privacy_trainer.train_with_privacy(
                self.model,
                batch['data'],
                self.quantum_optimizer
            )
            metrics.update(privacy_metrics)
            
            # Verify quantum security
            security_metrics = SECURITY_PROOFS.verify_quantum_security(
                self.model,
                batch['data']
            )
            metrics.update({
                f'security_{k}': v
                for k, v in security_metrics.items()
            })
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Complete training procedure"""
        history = defaultdict(list)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Update history
            for k, v in {**train_metrics, **val_metrics}.items():
                history[k].append(v)
            
            # Model saving
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if save_path:
                    self.save_checkpoint(save_path, epoch, best_val_loss)
            
            # Log progress
            self._log_progress(epoch, train_metrics, val_metrics)
        
        return history
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['data'])
                
                # Compute metrics
                loss = F.cross_entropy(outputs, batch['labels'])
                metrics['val_loss'] += loss.item()
                
                # Compute additional metrics
                preds = outputs.argmax(dim=1)
                metrics['val_acc'] += (preds == batch['labels']).float().mean().item()
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(self.val_loader)
        
        return metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float
    ) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.quantum_optimizer.optimizer.state_dict(),
            'val_loss': val_loss,
            'privacy_budget': self.privacy_trainer.epsilon_budget,
            'accumulated_cost': self.privacy_trainer.accumulated_cost
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.quantum_optimizer.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.privacy_trainer.epsilon_budget = checkpoint['privacy_budget']
        self.privacy_trainer.accumulated_cost = checkpoint['accumulated_cost']
    
    def _log_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log training progress"""
        metrics = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **val_metrics
        }
        
        # Log to console
        metrics_str = ' '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Log to wandb if configured
        if CONFIG.training.use_wandb:
            wandb.log(metrics)

# Create factory function for trainer
def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    distributed: bool = True
) -> QuantumResistantTrainer:
    """Create quantum-resistant trainer instance"""
    return QuantumResistantTrainer(
        model,
        train_loader,
        val_loader,
        device,
        distributed
    )