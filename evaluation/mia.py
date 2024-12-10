import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
import random

def warmup_cuda():
    dummy_tensor = torch.ones(1, device='cuda')
    del dummy_tensor
    torch.cuda.synchronize()

def compute_losses(model, loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_size = len(loader.dataset)
    all_losses = torch.empty(total_size, device='cpu')
    
    optimal_batch_size = loader.batch_size
    model.eval()
    start_idx = 0
    
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.no_grad(), torch.cuda.amp.autocast():
            for inputs, labels in loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                losses = criterion(outputs, labels)
                all_losses[start_idx:start_idx + batch_size] = losses.cpu()
                start_idx += batch_size
                stream.synchronize()
    
    torch.cuda.synchronize()
    return all_losses.numpy()

def compute_losses_fast(model, loader, device, max_samples=300):
    criterion = nn.CrossEntropyLoss(reduction="none")
    dataset_size = len(loader.dataset)
    
    if dataset_size > max_samples:
        # ndices = list(range(max_samples))
        indices = sorted(random.sample(range(dataset_size), max_samples))
        sampled_loader = DataLoader(
            Subset(loader.dataset, indices),
            batch_size=256,
            shuffle=False,
            pin_memory=True
        )
    else:
        sampled_loader = loader

    total_size = len(sampled_loader.dataset)
    all_losses = torch.empty(total_size, device='cpu')
    
    model.eval()
    start_idx = 0
    
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.no_grad(), torch.cuda.amp.autocast():
            for inputs, labels in sampled_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                losses = criterion(outputs, labels)
                all_losses[start_idx:start_idx + batch_size] = losses.cpu()
                start_idx += batch_size
                stream.synchronize()
    
    torch.cuda.synchronize()
    return all_losses.numpy()

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    
    attack_model = LogisticRegression(max_iter=1000)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def MIA(model, retain_loader, forget_loader, test_loader, device):
    warmup_cuda()

    optimal_batch_size = max(loader.batch_size for loader in [retain_loader, forget_loader, test_loader])
    for loader in [retain_loader, forget_loader, test_loader]:
        if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
            loader.batch_sampler.batch_size = optimal_batch_size
    
    retain_losses = compute_losses(model, retain_loader, device)
    forget_losses = compute_losses(model, forget_loader, device)
    test_losses = compute_losses(model, test_loader, device)
    
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = np.concatenate([np.zeros(len(test_losses)), np.ones(len(forget_losses))])
    
    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())
    
    return {
        'MIA CV Accuracy': float(mia_scores.mean()),
        'Forgetting Score': float(forgetting_score)
    }

def MIA_training_monitor(model, retain_loader, forget_loader, test_loader, device, max_samples=300):
    retain_losses = compute_losses_fast(model, retain_loader, device, max_samples)
    forget_losses = compute_losses_fast(model, forget_loader, device, max_samples)
    test_losses = compute_losses_fast(model, test_loader, device, max_samples)
    
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = np.concatenate([np.zeros(len(test_losses)), np.ones(len(forget_losses))])
    
    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())
    
    return {
        'MIA CV Accuracy': float(mia_scores.mean()),
        'Forgetting Score': float(forgetting_score)
    }
