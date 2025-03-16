import torch.nn as nn
import torchvision.models as models

class PlasmaCNN(nn.Module):
    """
    PlasmaCNN model built on ResNet-18, modified for regression tasks.
    
    Attributes:
        base_model (ResNet-18): Pre-trained ResNet-18 model with a modified fully connected layer.
    """
    
    def __init__(self):
        """
        Initializes the PlasmaCNN model by loading ResNet-18 and modifying the final layer.
        """
        super(PlasmaCNN, self).__init__()
        
        # Load a pre-trained ResNet-18 model as the base
        self.base_model = models.resnet18(pretrained=True)
        
        # Modify the fully connected layer for regression output
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1),  # Single output neuron for regression
            nn.ReLU()  # Ensures the output is non-negative
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (Tensor): Input image tensor.
        
        Returns:
            Tensor: Model output (continuous value for regression).
        """
        return self.base_model(x)