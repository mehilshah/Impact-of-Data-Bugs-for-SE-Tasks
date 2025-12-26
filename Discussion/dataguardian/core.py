import numpy as np
import torch
from scipy.stats import skew, kurtosis
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple
from .config import ThresholdConfig, DEFAULT_THRESHOLDS

class DataHealthMonitor:
    """Monitors model parameters and gradients for health indicators."""
    
    def __init__(self, model: torch.nn.Module, thresholds: Optional[ThresholdConfig] = None):
        self.model = model
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.metrics = defaultdict(OrderedDict)
        self.gradient_chain: List[Tuple[str, float]] = []
        self.layer_order: List[str] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register backward hooks for all parameters."""
        for name, param in self.model.named_parameters():
            self.layer_order.append(name)
            param.register_hook(self._create_hook(name))

    def _create_hook(self, name: str):
        """Create gradient hook that prepends to maintain forward order."""
        def hook_fn(grad: torch.Tensor) -> torch.Tensor:
            self._store_grad_metrics(name, grad)
            return grad
        return hook_fn

    def _compute_distribution_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive statistics for a tensor."""
        arr = tensor.detach().cpu().numpy().flatten()
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'var': float(np.var(arr)),
            'skewness': float(skew(arr)),
            'kurtosis': float(kurtosis(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'dead_ratio': float(np.mean(np.abs(arr) < 1e-7)),
            'l2_norm': float(np.linalg.norm(arr)),
            'q1': float(np.quantile(arr, 0.25)),
            'median': float(np.median(arr)),
            'q3': float(np.quantile(arr, 0.75))
        }

    def _store_grad_metrics(self, name: str, grad: torch.Tensor) -> None:
        """Store gradient metrics prepending to maintain forward order."""
        if grad is not None:
            self.metrics[name]['gradients'] = self._compute_distribution_stats(grad)
            # Prepend to maintain forward order in backward pass
            self.gradient_chain.insert(0, (name, self.metrics[name]['gradients']['l2_norm']))

    def analyze_parameters(self) -> None:
        """Analyze current model parameters."""
        for name, param in self.model.named_parameters():
            stats = {}
            if 'weight' in name:
                stats['weights'] = self._compute_distribution_stats(param.data)
            elif 'bias' in name:
                stats['biases'] = self._compute_distribution_stats(param.data)
            self.metrics[name].update(stats)

    def check_issues(self) -> List[Dict[str, any]]:
        """Check for all types of issues."""
        issues = []
        issues += self._check_layer_issues()
        issues += self._check_cross_layer_issues()
        return issues

    def _check_layer_issues(self) -> List[Dict[str, any]]:
        """Check for layer-specific issues."""
        issues = []
        for name in self.layer_order:
            layer_metrics = self.metrics.get(name, {})
            if not layer_metrics:
                continue

            if 'weights' in layer_metrics:
                w = layer_metrics['weights']
                if w['mean'] < self.thresholds.weight_mean_low:
                    issues.append(self._create_issue(name, 'Reduced Capacity', 'weight_mean', w['mean']))
                if w['var'] > self.thresholds.weight_var_high_code:
                    issues.append(self._create_issue(name, 'Concept Drift', 'weight_variance', w['var']))
                if abs(w['skewness']) > self.thresholds.weight_skew_thresh:
                    issues.append(self._create_issue(name, 'Distorted Weights', 'weight_skewness', w['skewness']))
                if abs(w['kurtosis']) > self.thresholds.kurtosis_thresh:
                    issues.append(self._create_issue(name, 'Weight Kurtosis', 'weight_kurtosis', w['kurtosis']))

            if 'biases' in layer_metrics:
                b = layer_metrics['biases']
                if b['mean'] < self.thresholds.bias_low:
                    issues.append(self._create_issue(name, 'Label Noise', 'bias_mean', b['mean']))
                if b['median'] > self.thresholds.bias_high:
                    issues.append(self._create_issue(name, 'Overfitting', 'bias_median', b['median']))
                if abs(b['skewness']) > self.thresholds.bias_skew_thresh:
                    issues.append(self._create_issue(name, 'Bias Skewness', 'bias_skewness', b['skewness']))

            if 'gradients' in layer_metrics:
                g = layer_metrics['gradients']
                if g['max'] - g['min'] > self.thresholds.grad_range_extreme:
                    issues.append(self._create_issue(name, 'Gradient Instability', 'grad_range', g['max'] - g['min']))
                if g['l2_norm'] < self.thresholds.grad_vanishing_thresh:
                    issues.append(self._create_issue(name, 'Vanishing Gradients', 'grad_norm', g['l2_norm']))
                if g['l2_norm'] > self.thresholds.grad_exploding_thresh:
                    issues.append(self._create_issue(name, 'Exploding Gradients', 'grad_norm', g['l2_norm']))
                if abs(g['skewness']) > self.thresholds.grad_skew_high:
                    issues.append(self._create_issue(name, 'Gradient Skew', 'grad_skewness', g['skewness']))
                if g['dead_ratio'] > self.thresholds.dead_neuron_ratio:
                    issues.append(self._create_issue(name, 'Sparse Updates', 'dead_neurons', g['dead_ratio']))

        return issues

    def _check_cross_layer_issues(self) -> List[Dict[str, any]]:
        """Check for issues between layers."""
        issues = []
        if len(self.gradient_chain) >= 2:
            input_layer, input_grad = self.gradient_chain[0]
            output_layer, output_grad = self.gradient_chain[-1]
            
            if output_grad > 0:
                grad_ratio = input_grad / output_grad
                if grad_ratio < 1/self.thresholds.layer_grad_ratio:
                    issues.append(self._create_cross_issue(
                        'Vanishing Gradients', grad_ratio, input_layer, output_layer))
                elif grad_ratio > self.thresholds.layer_grad_ratio:
                    issues.append(self._create_cross_issue(
                        'Exploding Gradients', grad_ratio, input_layer, output_layer))
        return issues

    def reset(self) -> None:
        """Reset monitoring state."""
        self.metrics.clear()
        self.gradient_chain.clear()

    def _create_issue(self, layer: str, issue_type: str, metric: str, value: float) -> Dict[str, any]:
        return {
            'type': issue_type,
            'metric': metric,
            'value': value,
            'layer': layer
        }

    def _create_cross_issue(self, issue_type: str, ratio: float, layer1: str, layer2: str) -> Dict[str, any]:
        return {
            'type': issue_type,
            'metric': 'gradient_ratio',
            'value': ratio,
            'layers': f"{layer1} to {layer2}"
        }