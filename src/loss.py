import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice loss for multi-class 3D segmentation.
    Computes smoothed dice coefficient for each class and returns the average loss.
    """
    def __init__(self, smooth=1e-6, num_classes=6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, target):
        """
        logits: (Batch, C, D, H, W)
        target: (Batch, D, H, W)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode the target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        # Reshape one-hot: (Batch, D, H, W, C) -> (Batch, C, D, H, W)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).contiguous()

        # Compute Dice for each class
        dims = (2, 3, 4) # Spatial dimensions
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs + target_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Loss mask: maybe exclude background if requested? 
        # Standard implementation averages overall classes.
        return 1 - dice_score.mean()

class HybridLoss(nn.Module):
    """
    Combined Loss = 0.5 * CrossEntropy + 0.5 * DiceLoss
    Handles class imbalance and provides smooth gradient flow.
    """
    def __init__(self, weight=None, num_classes=6):
        super(HybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes=num_classes)
        
    def forward(self, logits, target):
        return 0.5 * self.ce(logits, target) + 0.5 * self.dice(logits, target)

if __name__ == "__main__":
    # Test execution
    num_classes = 6
    loss_fn = HybridLoss(num_classes=num_classes)
    
    # Dummy outputs: (Batch=1, Classes=6, D=128, H=128, W=128)
    dummy_logits = torch.randn(1, 6, 128, 128, 128)
    dummy_target = torch.randint(0, 6, (1, 128, 128, 128))
    
    loss = loss_fn(dummy_logits, dummy_target)
    print(f"Hybrid Loss value: {loss.item():.4f}")
