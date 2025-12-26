from typing import Dict, List, Optional, Tuple
from .core import DataHealthMonitor
from .config import ThresholdConfig

class HealthCallback:
    """Integration with training frameworks for health monitoring."""
    
    def __init__(self, model: torch.nn.Module, thresholds: Optional[ThresholdConfig] = None,
                 layer_outlier_threshold: float = 70.0):
        self.monitor = DataHealthMonitor(model, thresholds)
        self.layer_outlier_thresh = layer_outlier_threshold

    def on_epoch_end(self) -> Tuple[bool, str]:
        """Trigger health checks at end of epoch."""
        self.monitor.analyze_parameters()
        issues = self.monitor.check_issues()
        significant, report = self._evaluate_issues(issues)
        self.monitor.reset()
        return not significant, report

    def _evaluate_issues(self, issues: List[Dict[str, any]]) -> Tuple[bool, str]:
        """Determine if issues are significant."""
        issue_counts = defaultdict(int)
        total_layers = len(set(name.split('.')[0] for name in self.monitor.layer_order))

        for issue in issues:
            issue_counts[issue['type']] += 1

        significant_issues = {
            issue: count for issue, count in issue_counts.items()
            if (count / total_layers) * 100 >= self.layer_outlier_thresh
        }

        if significant_issues:
            report = self._generate_report(significant_issues, issues)
            return True, report
        return False, "No significant issues detected"

    def _generate_report(self, significant_issues: Dict[str, int], issues: List[Dict[str, any]]) -> str:
        """Generate detailed diagnostic report."""
        report = ["\n=== Data Health Alert ==="]
        detailed_issues = defaultdict(list)

        for issue in issues:
            if issue['type'] in significant_issues:
                detailed_issues[issue['type']].append(issue)

        for issue_type, instances in detailed_issues.items():
            report.append(f"\n{issue_type} Detected in {significant_issues[issue_type]} layers:")
            for instance in instances:
                report.append(f"  - Layer {instance['layer']}: {instance['metric']} = {instance['value']:.2e}")

        # Add remediation advice
        remediation = {
            'Vanishing Gradients': "Try gradient clipping, batch normalization, or learning rate reduction",
            'Exploding Gradients': "Implement gradient clipping, weight regularization, or learning rate reduction",
            'Label Noise': "Apply data cleaning, label smoothing, or robust loss functions",
            'Concept Drift': "Consider online learning approaches or periodic model retraining",
            'Overfitting': "Add dropout, regularization, or increase training data diversity",
            'Reduced Capacity': "Check for data preprocessing issues, increase model capacity",
            'Gradient Instability': "Normalize input data, check for outliers in training data"
        }

        report.append("\nRecommended Actions:")
        for issue in significant_issues:
            if issue in remediation:
                report.append(f"  - {issue}: {remediation[issue]}")

        return "\n".join(report)