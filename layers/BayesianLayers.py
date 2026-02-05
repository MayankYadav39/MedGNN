import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer using Bayes by Backprop (Variational Inference).
    Weights are sampled from a Gaussian distribution.
    """
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Prior parameters
        self.prior_sigma = prior_sigma
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_logvar.data.fill_(-5.0)  # Small initial variance
        
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_logvar.data.fill_(-5.0)

    def forward(self, x, sample=True):
        if self.training or sample:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            eps_w = torch.randn_like(weight_std)
            weight = self.weight_mu + eps_w * weight_std
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            eps_b = torch.randn_like(bias_std)
            bias = self.bias_mu + eps_b * bias_std
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        Compute KL Divergence between the variational distribution and the prior.
        Assumes a Gaussian prior N(0, prior_sigma^2).
        """
        # KL(q(w)||p(w)) for weights
        weight_var = torch.exp(self.weight_logvar)
        kl_w = 0.5 * torch.sum(
            2 * math.log(self.prior_sigma) - self.weight_logvar + 
            (weight_var + self.weight_mu**2) / (self.prior_sigma**2) - 1
        )
        
        # KL(q(b)||p(b)) for bias
        bias_var = torch.exp(self.bias_logvar)
        kl_b = 0.5 * torch.sum(
            2 * math.log(self.prior_sigma) - self.bias_logvar + 
            (bias_var + self.bias_mu**2) / (self.prior_sigma**2) - 1
        )
        
        return kl_w + kl_b

class AleatoricLinear(nn.Module):
    """
    A linear layer that outputs both Mean (logits) and Log-Variance for aleatoric uncertainty.
    """
    def __init__(self, in_features, out_features):
        super(AleatoricLinear, self).__init__()
        # Outputs 2 * out_features: [logits, log_var]
        self.linear = nn.Linear(in_features, out_features * 2)
        self.out_features = out_features

    def forward(self, x):
        output = self.linear(x)
        # Split into mean and log-variance
        logits = output[..., :self.out_features]
        log_var = output[..., self.out_features:]
        return logits, log_var
