import torch
import torch.nn as nn
import torch.optim as optim  
from torch.optim.lr_scheduler import CosineAnnealingLR

def euk_unlearn(model, retain_loader, device, epochs, lr=0.001, momentum=0.9,  model_type='resnet'):
    
    for param in model.parameters():
        param.requires_grad = False

    if model_type == 'resnet':
        for name, param in model.named_parameters():
            if 'linear' in name or 'bn2' in name or 'final' in name or 'dropout' in name:
                param.requires_grad = True
    elif model_type == 'efficientnet':
        for name, param in model.named_parameters():
            if not ("stem_conv" in name or "layers" in name or "head_conv" in name):
                param.requires_grad = True
    elif model_type == 'densenet':
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name or 'bn' in name or 'conv' in name:
                param.requires_grad = True
    else:
        raise ValueError("Unsupported model type: Choose from 'resnet', 'efficientnet', or 'densenet'.")

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Please check your model and unfreezing logic.")
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()

        scheduler.step()
        
        avg_loss = total_loss / len(retain_loader.dataset)
        accuracy = 100. * total_correct / len(retain_loader.dataset)

        print(f"Unlearning Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model