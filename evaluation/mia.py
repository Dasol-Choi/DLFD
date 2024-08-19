import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

def compute_losses(model, loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, labels).cpu().numpy()
            all_losses.extend(losses)
    
    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = LogisticRegression()
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def cal_mia(model, forget_dataloader_test,unseen_dataloader, device):
    forget_losses = compute_losses(model, forget_dataloader_test, device)
    unseen_losses = compute_losses(model, unseen_dataloader, device)

    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(unseen_losses)]

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}


def MIA(model, retain_loader, forget_loader, test_loader, device):
    retain_losses = compute_losses(model, retain_loader, device)
    forget_losses = compute_losses(model, forget_loader, device)
    test_losses = compute_losses(model, test_loader, device)

    retain_labels = [0] * len(retain_losses)
    forget_labels = [1] * len(forget_losses)
    test_labels = [2] * len(test_losses)

    all_losses = np.concatenate([retain_losses, forget_losses, test_losses])
    all_labels = np.concatenate([retain_labels, forget_labels, test_labels])

    # Logistic regression MIA
    logistic_regression = LogisticRegression()
    logistic_regression.fit(all_losses.reshape(-1, 1), all_labels)
    predictions = logistic_regression.predict(all_losses.reshape(-1, 1))
    mia_accuracy = accuracy_score(all_labels, predictions)

    # Simple MIA with cross-validation
    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)
    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {
        'MIA Regression Accuracy': float(mia_accuracy),
        'MIA CV Accuracy': float(mia_scores.mean()),
        'Forgetting Score': float(forgetting_score)
    }
