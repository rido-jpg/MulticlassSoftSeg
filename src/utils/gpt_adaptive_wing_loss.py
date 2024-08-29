import torch
import torch.nn as nn

class AdaptiveWingLoss(nn.Module):
    def __init__(self, theta=0.5, alpha=2.1, omega=14, epsilon=1):
        """
        Initializes the Adaptive Wing Loss.
        
        Args:
            theta (float): Threshold between linear and non-linear loss.
            alpha (float): Controls the curvature of the non-linear part.
            omega (float): Multiplicative factor for the non-linear part of the loss.
            epsilon (float): Small value to prevent division by zero or numerical instability.
        """
        super(AdaptiveWingLoss, self).__init__()
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Forward pass to compute the Adaptive Wing Loss.
        
        Args:
            y_pred (torch.Tensor): Predicted heatmap.
            y_true (torch.Tensor): Ground truth heatmap.
            
        Returns:
            torch.Tensor: Computed loss.
        """
        delta_y = (y_true - y_pred).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]

        # Compute loss for delta_y < theta
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.epsilon, self.alpha - y_true[delta_y < self.theta]))
        
        # Compute loss for delta_y >= theta
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y_true)))
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y_true))
        loss2 = A * delta_y2 - C[delta_y >= self.theta]

        return (loss1.sum() + loss2.sum()) / (y_true.numel())