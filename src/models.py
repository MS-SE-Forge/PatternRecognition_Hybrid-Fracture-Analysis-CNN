import torch
import torch.nn as nn
import torchvision.models as models

class FractureCNN(nn.Module):
    """
    Base CNN model for Fracture Detection.
    Supports ResNet50 (default) and ResNet18 backbones.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(FractureCNN, self).__init__()
        
        self.backbone_name = backbone_name
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Initializing FractureCNN ({backbone_name}) on: {self.device}")

        # Initialize Backbone
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Modify first layer to accept grayscale (1 channel) instead of RGB (3)
        # ResNet conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fully connected layer for binary classification 
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2) 
        
        # Auto-move to the detected device
        self.to(self.device) 

    def forward(self, x):
        return self.backbone(x)

class EnsembleModel(nn.Module):
    """
    Simple averaging ensemble of two models.
    """
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.device = modelA.device # Assume both are on same device
        
    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        # return average of logits (soft voting)
        return (outA + outB) / 2.0
