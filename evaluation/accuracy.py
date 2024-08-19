import torch
import time

@torch.no_grad()
def calculate_accuracy(model, data_loader, criterion, device):
    start_time = time.time()
    model.eval()
    total = 0
    running_loss = 0.0
    running_corrects = 0
    running_top2_corrects = 0

    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Top-2 accuracy.
            _, top2_preds = outputs.topk(2, dim=1)
            top2_correct = top2_preds.eq(labels.view(-1, 1).expand_as(top2_preds))
            running_top2_corrects += top2_correct.any(dim=1).sum().item()

        total += labels.size(0)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    return {'Loss': running_loss / total, 'Acc': running_corrects / total, 'Top-2 Acc': running_top2_corrects / total}

def calculate_all_accuracies(model, data_loaders, criterion, device):
    accuracies = {}
    for key in ['retain', 'forget', 'test']:
        if key in data_loaders:
            accuracies[key] = calculate_accuracy(model, data_loaders[key], criterion, device)
            print(f"Model {key.capitalize()} Accuracy: {accuracies[key]['Acc']:.2f}%")
    return accuracies