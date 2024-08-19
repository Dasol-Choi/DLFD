import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)
        
    def forward(self):
        return self.noise


def generate_error_maximizing_noise(model, forgetloader, device, num_epochs=5, num_steps=8):
    
    forget_data = next(iter(forgetloader))
    forget_inputs, forget_labels = forget_data
    forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)

    noise = Noise(*forget_inputs.shape).to(device)
    noise_optimizer = torch.optim.Adam(noise.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = []
        for step in range(num_steps):
            noise_inputs = noise()
            outputs = model(noise_inputs)
            loss = -F.cross_entropy(outputs, forget_labels) + 0.1 * torch.mean(torch.sum(torch.square(noise_inputs), [1, 2, 3]))
            noise_optimizer.zero_grad()
            loss.backward()
            noise_optimizer.step()
            total_loss.append(loss.item())
            
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Noise Loss: {np.mean(total_loss):.4f}")

    return noise

def impair_step(model, noisy_loader, device, lr=0.001, impair_epochs=1):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    running_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(impair_epochs):
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            running_acc += (labels == out).sum().item()
            
        print(f"Impair Step - Train loss {epoch+1}: {running_loss/len(noisy_loader.dataset):.4f}, Train Acc: {running_acc*100/len(noisy_loader.dataset):.2f}%")


def repair_step(model, heal_loader, device, lr=0.001, repair_epochs=1):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    running_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(repair_epochs):
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            running_acc += (labels == out).sum().item()
            
        print(f"Repair Step - Train loss {epoch+1}: {running_loss/len(heal_loader.dataset):.4f}, Train Acc: {running_acc*100/len(heal_loader.dataset):.2f}%")

