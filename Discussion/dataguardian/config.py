from dataclasses import dataclass

@dataclass
class ThresholdConfig:
    """Configuration for monitoring thresholds."""
    # General thresholds
    bias_low: float = 0.01               # RQ1: Inconsistent learning
    bias_high: float = 1.0               # RQ2: Overfitting
    weight_mean_low: float = 0.05        # RQ1: Reduced capacity
    weight_var_high_code: float = 2.3    # RQ1: Concept drift (code)
    weight_var_high_text: float = 5.76   # RQ3: Metric data variance
    grad_skew_high: float = 1.5          # RQ2: Gradient skewness
    grad_range_extreme: float = 1e1      # RQ1: Gradient instability
    dead_neuron_ratio: float = 0.85      # RQ3: Sparse updates
    grad_vanishing_thresh: float = 1e-6  # RQ3: Vanishing gradients
    grad_exploding_thresh: float = 1e1   # RQ3: Exploding gradients
    bias_skew_thresh: float = 1.0        # RQ1: Distorted distributions
    weight_skew_thresh: float = 1.0      # RQ2: Abnormal weights
    kurtosis_thresh: float = 3.0         # General distribution shape
    layer_grad_ratio: float = 1e3        # RQ3: Vanishing gradient factor
    weight_spread_thresh: float = 2.0    # RQ2: Numerical instability
    convergence_stall: float = 1e-6      # RQ1: Slow convergence

# Default configuration
DEFAULT_THRESHOLDS = ThresholdConfig()