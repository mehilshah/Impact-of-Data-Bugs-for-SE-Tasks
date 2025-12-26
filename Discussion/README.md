## DataGuardian

A comprehensive health monitoring system for deep learning models that detects data quality issues during training by analyzing parameter distributions, gradients, and layer behaviors.

## Overview

DataGuardian helps you identify potential data quality issues early in the training process by monitoring statistical patterns in your model's parameters and gradients. The system analyzes various metrics including distribution characteristics (mean, variance, skewness, kurtosis), gradient flow, and layer-wise behaviors to detect common problems such as:

-   Vanishing or exploding gradients
-   Overfitting
-   Label noise
-   Concept drift
-   Gradient instability
-   Reduced model capacity
-   Sparse updates

## Quick Start

```python
import torch
from dataguardian import HealthCallback

# Initialize your model
model = YourModel()

# Create the health monitor callback
health_callback = HealthCallback(model)

# During training (e.g., after each epoch)
is_healthy, report = health_callback.on_epoch_end()

if not is_healthy:
    print(report)  # Displays detected issues and recommendations

```
## Core Features
-   **Continuous Monitoring**: Tracks model health throughout the training process
-   **Comprehensive Metrics**: Analyzes over 12 statistical indicators across weights, biases, and gradients
-   **Actionable Insights**: Provides specific recommendations to address detected issues
-   **Flexible Configuration**: Allows customization of monitoring thresholds
-   **Framework Agnostic**: Works with PyTorch models regardless of training framework

## Use Cases

### Detecting Data Quality Issues During Training

```python
from dataguardian import HealthCallback
import torch

# Initialize model and training components
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
health_callback = HealthCallback(model)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Check model health at end of epoch
    is_healthy, report = health_callback.on_epoch_end()
    if not is_healthy:
        print(f"Epoch {epoch}: Health issues detected")
        print(report)
        # Potential actions: decrease learning rate, check data, etc.

```

### Integration with Custom Training Loops

```python
from dataguardian import DataHealthMonitor
import torch

# Initialize model and training components
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
monitor = DataHealthMonitor(model)

# Custom training loop with gradient accumulation
for epoch in range(num_epochs):
    accumulated_issues = []
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Check health every N batches
        if batch_idx % 50 == 0:
            monitor.analyze_parameters()
            batch_issues = monitor.check_issues()
            accumulated_issues.extend(batch_issues)
            monitor.reset()
        
        optimizer.step()
        optimizer.zero_grad()
    
    # Analyze accumulated issues
    issue_types = {}
    for issue in accumulated_issues:
        issue_type = issue['type']
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    print(f"Epoch {epoch} health summary:")
    for issue_type, count in issue_types.items():
        print(f"- {issue_type}: {count} occurrences")
```

### Early Warning System for Production Models

```python
from dataguardian import HealthCallback
import torch
import time

# Initialize model and monitoring components
model = ProductionModel()
health_callback = HealthCallback(model, layer_outlier_threshold=50.0)  # More sensitive threshold

# Monitoring loop for production model
while True:
    # Get latest batch of production data
    inputs, targets = get_production_data_batch()
    
    # Run forward and backward pass to collect metrics
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Check model health
    is_healthy, report = health_callback.on_epoch_end()
    
    if not is_healthy:
        # Alert the team
        send_alert("Model health issue detected: " + report)
        # Log detailed report
        log_issue(report)
    
    # Sleep before next check
    time.sleep(3600)  # Check hourly
```

## Customizing Thresholds
DataGuardian provides sensible default thresholds derived from empirical research, but you can customize them to match your specific use case:

```python
from dataguardian import DataHealthMonitor, ThresholdConfig

# Create custom threshold configuration
custom_thresholds = ThresholdConfig(
    # Customize threshold values based on your domain knowledge
    grad_vanishing_thresh=1e-7,  # More sensitive to vanishing gradients
    dead_neuron_ratio=0.75,      # Adjusted for your activation patterns
    layer_grad_ratio=1e4,        # Higher tolerance for gradient differences
    weight_spread_thresh=1.5     # Tighter constraint on weight distributions
)

# Create monitor with custom thresholds
monitor = DataHealthMonitor(model, thresholds=custom_thresholds)
```

## How It Works

DataGuardian operates by:

1.  **Hooking into model parameters**: Registers backward hooks to capture gradient flow
2.  **Computing statistical metrics**: Analyzes distribution patterns for weights, biases, and gradients
3.  **Applying thresholds**: Compares metrics to empirically determined or custom thresholds
4.  **Aggregating issues**: Groups and evaluates issues across layers to determine significance
5.  **Providing remediation**: Offers targeted recommendations for detected problems