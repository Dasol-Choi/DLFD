import torch
import torch.optim as optim
from evaluation.accuracy import calculate_accuracy

def neggrad_unlearn(model, forget_loader, criterion, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs): 
        running_loss = 0.0
        for i, data in enumerate(forget_loader, 0):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            (-loss).backward()  
            optimizer.step()
            running_loss += loss.item()
        print(f"NegGrad Epoch [{epoch + 1}/{args.epochs}] Loss: {running_loss / len(forget_loader):.3f}")

