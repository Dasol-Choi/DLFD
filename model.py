import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        
        self.base_model = models.resnet18(pretrained=False)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # Define the new architecture with additional layers
        self.features = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Extract features using the base ResNet18 model
        x = self.base_model(x)
        
        # Make sure to flatten the output from the base model
        if len(x.shape) > 2:  # Flatten if it has more than two dimensions
            x = torch.flatten(x, start_dim=1)
        
        # Pass the flattened features through the additional layers
        x = self.features(x)
        return x

def load_model(num_classes, device, model_path=None):
    model = ResNet(num_classes)
    model = model.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model
