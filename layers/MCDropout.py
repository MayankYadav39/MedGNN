import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that stays active during both training and inference.
    This enables Bayesian approximation for uncertainty estimation.
    """
    def __init__(self, p=0.1):
        super(MCDropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        # Always apply dropout, even during eval mode
        # This is the key difference from standard nn.Dropout
        return F.dropout(x, p=self.p, training=True)


def compute_mc_predictions(model, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, mc_samples=10):
    """
    Run multiple forward passes with MC Dropout to get prediction distribution.
    
    Args:
        model: The neural network model
        x_enc, x_mark_enc, x_dec, x_mark_dec: Input tensors
        mask: Optional attention mask
        mc_samples: Number of MC sampling iterations
    
    Returns:
        mean_pred: Mean prediction across MC samples (B, num_class)
        uncertainty: Predictive uncertainty (B,) - variance across samples
        all_preds: All predictions (mc_samples, B, num_class)
    """
    model.eval()  # Set to eval mode (but MC Dropout will still be active)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(mc_samples):
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            predictions.append(output)
    
    # Stack predictions: (mc_samples, B, num_class)
    all_preds = torch.stack(predictions, dim=0)
    
    # Compute mean and variance
    mean_pred = all_preds.mean(dim=0)  # (B, num_class)
    
    # For classification, we can use:
    # 1. Variance of logits
    # 2. Predictive entropy
    # 3. Mutual information
    
    # Variance across MC samples (mean across classes)
    var_pred = all_preds.var(dim=0).mean(dim=1)  # (B,)
    
    return mean_pred, var_pred, all_preds


def compute_predictive_entropy(all_preds):
    """
    Compute predictive entropy as an uncertainty measure.
    
    Args:
        all_preds: Predictions from MC samples (mc_samples, B, num_class)
    
    Returns:
        entropy: Predictive entropy for each sample (B,)
    """
    # Convert logits to probabilities
    probs = F.softmax(all_preds, dim=-1)  # (mc_samples, B, num_class)
    
    # Average probabilities across MC samples
    mean_probs = probs.mean(dim=0)  # (B, num_class)
    
    # Compute entropy: -sum(p * log(p))
    epsilon = 1e-10  # For numerical stability
    entropy = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=1)  # (B,)
    
    return entropy


def compute_mutual_information(all_preds):
    """
    Compute mutual information between predictions and model parameters.
    This captures epistemic (model) uncertainty.
    
    Args:
        all_preds: Predictions from MC samples (mc_samples, B, num_class)
    
    Returns:
        mutual_info: Mutual information for each sample (B,)
    """
    # Convert to probabilities
    probs = F.softmax(all_preds, dim=-1)  # (mc_samples, B, num_class)
    
    # Predictive entropy (total uncertainty)
    predictive_entropy = compute_predictive_entropy(all_preds)
    
    # Expected entropy (aleatoric uncertainty)
    epsilon = 1e-10
    individual_entropy = -(probs * torch.log(probs + epsilon)).sum(dim=-1)  # (mc_samples, B)
    expected_entropy = individual_entropy.mean(dim=0)  # (B,)
    
    # Mutual information = predictive entropy - expected entropy
    mutual_info = predictive_entropy - expected_entropy
    
    return mutual_info


def get_uncertainty_metrics(all_preds):
    """
    Compute multiple uncertainty metrics from MC predictions.
    
    Args:
        all_preds: Predictions from MC samples (mc_samples, B, num_class)
    
    Returns:
        Dictionary with uncertainty metrics
    """
    mean_pred = all_preds.mean(dim=0)
    var_pred = all_preds.var(dim=0).mean(dim=1)
    predictive_entropy = compute_predictive_entropy(all_preds)
    mutual_info = compute_mutual_information(all_preds)
    
    return {
        'mean_prediction': mean_pred,
        'variance': var_pred,
        'predictive_entropy': predictive_entropy,
        'mutual_information': mutual_info,
    }
