import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer import train, test  

def cfk_train(model, train_loader, test_loader, device, epochs, lr=0.001, momentum=0.9, model_type='resnet'):
    if not isinstance(lr, float):
        raise TypeError(f"Expected lr to be float, got {type(lr)}")
    if not isinstance(momentum, float):
        raise TypeError(f"Expected momentum to be float, got {type(momentum)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Freeze layers based on model type
    if model_type == 'resnet':
        # Freeze all base model parameters for ResNet
        for name, param in model.named_parameters():
            if "base_model" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        # Print metrics for this epoch
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Epoch [{epoch+1}/{epochs}] - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Step the scheduler
        scheduler.step()

    return model
