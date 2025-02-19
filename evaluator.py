import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, confusion_matrix
)
import logging
from config import CONFIG
import time
from security_proofs import SecurityProofs
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast

class QuantumSecurityEvaluator:
    """Quantum security evaluation and metrics"""
    
    def __init__(self):
        self.quantum_metrics = self._initialize_quantum_metrics()
        self.security_proofs = SecurityProofs()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_quantum_metrics(self) -> Dict[str, Any]:
        """Initialize quantum security metrics"""
        return {
            "grover_resistance": 0.0,
            "shor_resistance": 0.0,
            "post_quantum_security": 0.0,
            "quantum_bit_security": 0.0,
            "lattice_security": 0.0
        }
    
    def evaluate_quantum_resistance(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate quantum security of the model"""
        metrics = {}
        
        # Evaluate Grover's algorithm resistance
        grover_metrics = self._evaluate_grover_resistance(model, test_data)
        metrics.update(grover_metrics)
        
        # Evaluate Shor's algorithm resistance
        shor_metrics = self._evaluate_shor_resistance(model)
        metrics.update(shor_metrics)
        
        # Evaluate post-quantum cryptography
        pqc_metrics = self._evaluate_post_quantum_security(model)
        metrics.update(pqc_metrics)
        
        # Calculate overall quantum security score
        metrics["overall_quantum_security"] = self._compute_overall_security(metrics)
        
        return metrics
    
    def _evaluate_grover_resistance(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate resistance against Grover's algorithm"""
        param_space = sum(p.numel() for p in model.parameters())
        grover_iterations = int(np.sqrt(param_space))
        
        # Simulate quantum search attack
        success_prob = 1 - np.exp(-grover_iterations / np.sqrt(param_space))
        
        return {
            "grover_resistance": 1 - success_prob,
            "search_space_size": param_space,
            "estimated_grover_iterations": grover_iterations
        }
    
    def _evaluate_shor_resistance(
        self,
        model: torch.nn.Module
    ) -> Dict[str, float]:
        """Evaluate resistance against Shor's algorithm"""
        # Evaluate cryptographic components
        key_strength = self.security_proofs.evaluate_key_strength(model)
        factorization_resistance = self.security_proofs.evaluate_factorization_resistance()
        
        return {
            "shor_resistance": min(key_strength, factorization_resistance),
            "key_strength": key_strength,
            "factorization_resistance": factorization_resistance
        }
    
    def _evaluate_post_quantum_security(
        self,
        model: torch.nn.Module
    ) -> Dict[str, float]:
        """Evaluate post-quantum cryptographic security"""
        lattice_security = self._evaluate_lattice_security(model)
        quantum_bit_security = self._compute_quantum_bit_security(model)
        
        return {
            "post_quantum_security": min(lattice_security, quantum_bit_security),
            "lattice_security": lattice_security,
            "quantum_bit_security": quantum_bit_security
        }
    
    def _evaluate_lattice_security(self, model: torch.nn.Module) -> float:
        """Evaluate lattice-based security"""
        lattice_dim = CONFIG.quantum.lattice_dimension
        noise_ratio = CONFIG.quantum.post_quantum_params["noise_width"]
        
        # Compute security based on lattice parameters
        security_level = np.log2(lattice_dim * noise_ratio)
        return min(1.0, security_level / 256.0)
    
    def _compute_quantum_bit_security(self, model: torch.nn.Module) -> float:
        """Compute quantum bit security level"""
        total_params = sum(p.numel() for p in model.parameters())
        return min(1.0, np.log2(total_params) / 256.0)
    
    def _compute_overall_security(self, metrics: Dict[str, float]) -> float:
        """Compute overall quantum security score"""
        weights = {
            "grover_resistance": 0.3,
            "shor_resistance": 0.3,
            "post_quantum_security": 0.4
        }
        return sum(metrics[k] * weights[k] for k in weights)

class Layer2Evaluator:
    """Layer-2 and rollup evaluation"""
    
    def __init__(self):
        self.rollup_verifier = RollupVerifier()
        self.cross_layer_metrics = CrossLayerMetrics()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_layer2_performance(
        self,
        model: torch.nn.Module,
        layer2_data: Dict[str, torch.Tensor],
        mainnet_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate Layer-2 performance"""
        metrics = {}
        
        # Evaluate rollup performance
        rollup_metrics = self.rollup_verifier.verify_rollups(layer2_data)
        metrics["rollup"] = rollup_metrics
        
        # Evaluate cross-layer consistency
        consistency_metrics = self.cross_layer_metrics.evaluate_consistency(
            layer2_data,
            mainnet_data
        )
        metrics["consistency"] = consistency_metrics
        
        # Evaluate Layer-2 specific metrics
        l2_metrics = self._evaluate_layer2_specific(model, layer2_data)
        metrics["layer2"] = l2_metrics
        
        return metrics
    
    def _evaluate_layer2_specific(
        self,
        model: torch.nn.Module,
        layer2_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate Layer-2 specific metrics"""
        metrics = {}
        
        # Throughput analysis
        metrics["tps"] = self._compute_throughput(layer2_data)
        
        # Latency analysis
        metrics["latency"] = self._compute_latency(model, layer2_data)
        
        # Cost analysis
        metrics["cost_efficiency"] = self._compute_cost_efficiency(layer2_data)
        
        return metrics
    
    def _compute_throughput(self, layer2_data: Dict[str, torch.Tensor]) -> float:
        """Compute transactions per second"""
        total_tx = sum(data.size(0) for data in layer2_data.values())
        time_window = CONFIG.model.layer2_config.batch_size / CONFIG.scalability.throughput_tps
        return total_tx / time_window
    
    def _compute_latency(
        self,
        model: torch.nn.Module,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Compute average latency"""
        start_time = time.time()
        with torch.no_grad():
            for data in layer2_data.values():
                _ = model(data)
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to ms
    
    def _compute_cost_efficiency(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Compute cost efficiency"""
        total_gas = sum(len(data) * CONFIG.model.layer2_config.max_batch_gas 
                       for data in layer2_data.values())
        return 1.0 / (1.0 + total_gas)

class CrossLayerMetrics:
    """Cross-layer consistency metrics"""
    
    def evaluate_consistency(
        self,
        layer2_data: Dict[str, torch.Tensor],
        mainnet_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate cross-layer consistency"""
        metrics = {}
        
        # Pattern consistency
        metrics["pattern_consistency"] = self._evaluate_pattern_consistency(
            layer2_data,
            mainnet_data
        )
        
        # State consistency
        metrics["state_consistency"] = self._evaluate_state_consistency(
            layer2_data,
            mainnet_data
        )
        
        # Transaction consistency
        metrics["tx_consistency"] = self._evaluate_transaction_consistency(
            layer2_data,
            mainnet_data
        )
        
        return metrics
    
    def _evaluate_pattern_consistency(
        self,
        layer2_data: Dict[str, torch.Tensor],
        mainnet_data: Dict[str, torch.Tensor]
    ) -> float:
        """Evaluate pattern consistency between layers"""
        consistencies = []
        for chain in layer2_data:
            if chain in mainnet_data:
                l2_patterns = self._extract_patterns(layer2_data[chain])
                l1_patterns = self._extract_patterns(mainnet_data[chain])
                
                consistency = F.cosine_similarity(
                    l2_patterns,
                    l1_patterns,
                    dim=0
                ).mean().item()
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _extract_patterns(self, data: torch.Tensor) -> torch.Tensor:
        """Extract patterns from data"""
        return F.normalize(data.view(data.size(0), -1), dim=1)
    
    def _evaluate_state_consistency(
        self,
        layer2_data: Dict[str, torch.Tensor],
        mainnet_data: Dict[str, torch.Tensor]
    ) -> float:
        """Evaluate state consistency between layers"""
        state_diffs = []
        for chain in layer2_data:
            if chain in mainnet_data:
                l2_state = layer2_data[chain].sum(dim=0)
                l1_state = mainnet_data[chain].sum(dim=0)
                
                state_diff = torch.norm(l2_state - l1_state).item()
                state_diffs.append(state_diff)
        
        return 1.0 / (1.0 + np.mean(state_diffs)) if state_diffs else 0.0
    
    def _evaluate_transaction_consistency(
        self,
        layer2_data: Dict[str, torch.Tensor],
        mainnet_data: Dict[str, torch.Tensor]
    ) -> float:
        """Evaluate transaction consistency between layers"""
        consistencies = []
        for chain in layer2_data:
            if chain in mainnet_data:
                l2_tx = layer2_data[chain]
                l1_tx = mainnet_data[chain]
                
                overlap = torch.min(l2_tx, l1_tx).sum()
                total = torch.max(l2_tx, l1_tx).sum()
                
                consistency = (overlap / total).item() if total > 0 else 0.0
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.0

class RollupVerifier:
    """Verify rollup correctness and performance"""
    
    def verify_rollups(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Verify rollup correctness and performance"""
        metrics = {}
        
        # Verify proof correctness
        metrics["proof_validity"] = self._verify_proofs(layer2_data)
        
        # Verify data availability
        metrics["data_availability"] = self._verify_data_availability(layer2_data)
        
        # Verify compression efficiency
        metrics["compression_ratio"] = self._compute_compression_ratio(layer2_data)
        
        # Verify batch efficiency
        metrics["batch_efficiency"] = self._compute_batch_efficiency(layer2_data)
        
        return metrics
    
    def _verify_proofs(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Verify zero-knowledge proofs"""
        valid_proofs = 0
        total_proofs = 0
        
        for data in layer2_data.values():
            proof = self._extract_proof(data)
            if proof is not None:
                total_proofs += 1
                if self._verify_single_proof(proof, data):
                    valid_proofs += 1
        
        return valid_proofs / total_proofs if total_proofs > 0 else 0.0
    
    def _verify_data_availability(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Verify data availability"""
        total_size = sum(data.numel() for data in layer2_data.values())
        available_size = sum(
            data.numel()
            for data in layer2_data.values()
            if self._check_availability(data)
        )
        
        return available_size / total_size if total_size > 0 else 0.0
    
    def _compute_compression_ratio(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Compute actual compression ratio"""
        original_size = sum(data.numel() * data.element_size() 
                          for data in layer2_data.values())
        compressed_size = sum(self._get_compressed_size(data) 
                            for data in layer2_data.values())
        
        return original_size / compressed_size if compressed_size > 0 else 0.0
    
    def _compute_batch_efficiency(
        self,
        layer2_data: Dict[str, torch.Tensor]
    ) -> float:
        """Compute batch processing efficiency"""
        total_batches = sum(1 for _ in layer2_data.values())
        efficient_batches = sum(
            1 for data in layer2_data.values()
            if self._is_batch_efficient(data)
        )
        
        return efficient_batches / total_batches if total_batches > 0 else 0.0

class PrivacyEvaluator:
    """Enhanced privacy evaluation with quantum considerations"""
    
    def __init__(self):
        self.quantum_evaluator = QuantumSecurityEvaluator()
        self.security_proofs = SecurityProofs()
        self.logger = logging.getLogger(__name__)
    
    # Continuing PrivacyEvaluator class...
    def evaluate_privacy_guarantees(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor,
        epsilon: float,
        delta: float,
        num_samples: int
    ) -> Dict[str, float]:
        """Evaluate comprehensive privacy guarantees"""
        try:
            # Quantum security evaluation
            quantum_metrics = self.quantum_evaluator.evaluate_quantum_resistance(
                model,
                test_data
            )
            
            # Differential privacy verification
            dp_metrics = self._verify_differential_privacy(
                model,
                epsilon,
                delta,
                num_samples
            )
            
            # Privacy loss computation
            privacy_loss = self._compute_privacy_loss(model, test_data)
            
            # Membership inference risk
            membership_risk = self._evaluate_membership_inference(model, test_data)
            
            return {
                **quantum_metrics,
                **dp_metrics,
                'privacy_loss': privacy_loss,
                'membership_risk': membership_risk,
                'epsilon_spent': self._compute_epsilon_spent(model),
                'remaining_budget': epsilon - self._compute_epsilon_spent(model)
            }
            
        except Exception as e:
            self.logger.error(f"Privacy evaluation failed: {str(e)}")
            return {}
    
    def _verify_differential_privacy(
        self,
        model: torch.nn.Module,
        epsilon: float,
        delta: float,
        num_samples: int
    ) -> Dict[str, float]:
        """Verify differential privacy guarantees"""
        params = {
            'noise_multiplier': CONFIG.model.noise_multiplier,
            'sensitivity': self._estimate_sensitivity(model),
            'num_queries': num_samples
        }
        
        # Compute DP guarantees
        dp_eps = self.security_proofs.verify_differential_privacy(params, delta)
        
        # Advanced composition theorem
        composed_eps = self._compute_composed_privacy(dp_eps, num_samples)
        
        return {
            'dp_epsilon': dp_eps,
            'composed_epsilon': composed_eps,
            'dp_delta': delta,
            'privacy_guarantee': dp_eps <= epsilon
        }
    
    def _compute_privacy_loss(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor
    ) -> float:
        """Compute empirical privacy loss"""
        with torch.no_grad():
            # Get model predictions
            outputs = model(test_data)
            
            # Compute privacy loss using moment accountant
            privacy_loss = self._compute_moment_accountant(outputs)
            
        return privacy_loss
    
    def _compute_moment_accountant(
        self,
        outputs: torch.Tensor,
        order: int = 32
    ) -> float:
        """Compute privacy loss using moment accountant method"""
        moments = []
        for k in range(1, order + 1):
            moment = self._compute_moment(outputs, k)
            moments.append(moment)
        
        return max(moments)
    
    def _compute_moment(
        self,
        outputs: torch.Tensor,
        order: int
    ) -> float:
        """Compute single moment for privacy analysis"""
        return (order * (order + 1) * outputs.var()) / (2 * CONFIG.model.noise_multiplier**2)
    
    def _evaluate_membership_inference(
        self,
        model: torch.nn.Module,
        test_data: torch.Tensor
    ) -> float:
        """Evaluate membership inference attack risk"""
        predictions = model(test_data)
        confidences = F.softmax(predictions, dim=1).max(dim=1)[0]
        
        # Compute membership inference vulnerability
        vulnerability = self._compute_membership_vulnerability(confidences)
        
        return vulnerability
    
    def _compute_membership_vulnerability(
        self,
        confidences: torch.Tensor
    ) -> float:
        """Compute membership inference vulnerability score"""
        # Higher confidence indicates higher vulnerability
        return float(confidences.mean())
    
    def _compute_epsilon_spent(
        self,
        model: torch.nn.Module
    ) -> float:
        """Compute spent privacy budget"""
        return model.privacy_layer.accumulated_cost if hasattr(model, 'privacy_layer') else 0.0
    
    def _estimate_sensitivity(
        self,
        model: torch.nn.Module
    ) -> float:
        """Estimate model's sensitivity"""
        param_norms = [p.norm(2).item() for p in model.parameters()]
        return max(param_norms)

class Evaluator:
    """Main evaluation class combining all components"""
    
    def __init__(self):
        self.privacy_evaluator = PrivacyEvaluator()
        self.quantum_evaluator = QuantumSecurityEvaluator()
        self.layer2_evaluator = Layer2Evaluator()
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        layer2_data: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation"""
        evaluation_results = {}
        
        # Quantum security evaluation
        quantum_results = self.quantum_evaluator.evaluate_quantum_resistance(
            model,
            next(iter(test_data.values()))  # Use first chain's data
        )
        evaluation_results['quantum'] = quantum_results
        
        # Privacy guarantees evaluation
        privacy_results = self.privacy_evaluator.evaluate_privacy_guarantees(
            model,
            next(iter(test_data.values())),
            CONFIG.model.privacy_epsilon,
            CONFIG.model.privacy_delta,
            sum(len(data) for data in test_data.values())
        )
        evaluation_results['privacy'] = privacy_results
        
        # Layer-2 evaluation if data is available
        if layer2_data is not None:
            layer2_results = self.layer2_evaluator.evaluate_layer2_performance(
                model,
                layer2_data,
                test_data
            )
            evaluation_results['layer2'] = layer2_results
        
        # Performance evaluation
        performance_results = self._evaluate_performance(model, test_data, device)
        evaluation_results['performance'] = performance_results
        
        return evaluation_results
    
    def _evaluate_performance(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model performance metrics"""
        metrics = {}
        
        # Compute accuracy metrics
        accuracy_metrics = self._compute_accuracy_metrics(
            model,
            test_data,
            device
        )
        metrics.update(accuracy_metrics)
        
        # Compute efficiency metrics
        efficiency_metrics = self._compute_efficiency_metrics(
            model,
            test_data,
            device
        )
        metrics.update(efficiency_metrics)
        
        # Compute scalability metrics
        scalability_metrics = self._compute_scalability_metrics(
            model,
            test_data
        )
        metrics.update(scalability_metrics)
        
        return metrics
    
    def _compute_accuracy_metrics(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """Compute accuracy-related metrics"""
        results = {}
        model.eval()
        
        with torch.no_grad(), autocast(enabled=CONFIG.training.mixed_precision):
            for chain, data in test_data.items():
                data = data.to(device)
                outputs = model(data)
                
                # Compute chain-specific metrics
                chain_metrics = {
                    f"{chain}_accuracy": (outputs.argmax(1) == data.y).float().mean().item(),
                    f"{chain}_auc": roc_auc_score(data.y.cpu(), outputs.softmax(1)[:, 1].cpu())
                }
                results.update(chain_metrics)
        
        # Compute average metrics
        results['avg_accuracy'] = np.mean([
            v for k, v in results.items() if k.endswith('accuracy')
        ])
        results['avg_auc'] = np.mean([
            v for k, v in results.items() if k.endswith('auc')
        ])
        
        return results
    
    def _compute_efficiency_metrics(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, float]:
        """Compute efficiency-related metrics"""
        # Measure inference time
        start_time = time.time()
        with torch.no_grad(), autocast(enabled=CONFIG.training.mixed_precision):
            for data in test_data.values():
                _ = model(data.to(device))
        inference_time = time.time() - start_time
        
        # Compute memory usage
        memory_usage = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        return {
            'inference_time_ms': inference_time * 1000 / len(test_data),
            'memory_usage_mb': memory_usage / 1024 / 1024,
            'throughput_samples_per_sec': len(test_data) / inference_time
        }
    
    def _compute_scalability_metrics(
        self,
        model: torch.nn.Module,
        test_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute scalability-related metrics"""
        total_params = sum(p.numel() for p in model.parameters())
        total_samples = sum(len(data) for data in test_data.values())
        
        return {
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'samples_processed': total_samples,
            'params_per_sample': total_params / total_samples
        }

    def generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("Enhanced HIPADual Evaluation Report")
        report.append("=" * 50)
        
        # Quantum Security Analysis
        report.append("\n1. Quantum Security Analysis")
        report.append("-" * 30)
        quantum_metrics = evaluation_results.get('quantum', {})
        for metric, value in quantum_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Privacy Analysis
        report.append("\n2. Privacy Analysis")
        report.append("-" * 30)
        privacy_metrics = evaluation_results.get('privacy', {})
        for metric, value in privacy_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Layer-2 Analysis
        if 'layer2' in evaluation_results:
            report.append("\n3. Layer-2 Analysis")
            report.append("-" * 30)
            layer2_metrics = evaluation_results['layer2']
            for category, metrics in layer2_metrics.items():
                report.append(f"\n{category.title()}:")
                for metric, value in metrics.items():
                    report.append(f"  {metric}: {value:.4f}")
        
        # Performance Analysis
        report.append("\n4. Performance Analysis")
        report.append("-" * 30)
        performance_metrics = evaluation_results.get('performance', {})
        for metric, value in performance_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        return "\n".join(report)