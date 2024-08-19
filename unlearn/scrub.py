import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evaluation.accuracy import calculate_accuracy

class Args:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.optim = "sgd"  # Options: 'sgd', 'adam', 'rmsp'
        self.sgda_epochs = 2
        self.sgda_learning_rate = 0.001
        self.sgda_weight_decay = 5e-4
        self.sgda_momentum = 0.9
        self.msteps = 2
        self.kd_T = 4  # Temperature for knowledge distillation
        # Learning Rate Decay
        self.lr_decay_epochs = [3, 5, 9]
        self.lr_decay_rate = 0.1

def sgda_adjust_learning_rate(epoch, args, optimizer):
    
    lr = args.sgda_learning_rate
    if epoch in args.lr_decay_epochs:
        lr *= args.lr_decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def train_distill(epoch, dataloader, module_list, optimizer, criterion_list, args, mode="minimize"):

    model_s = module_list[0]  # Assuming model_s is the student model
    model_t = module_list[1]  # Assuming model_t is the teacher model
    model_s.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()

        # Forward pass
        outputs_s = model_s(inputs)
        with torch.no_grad():
            outputs_t = model_t(inputs)

        # Compute losses
        loss_cls = criterion_list[0](outputs_s, labels)
        loss_div = criterion_list[1](outputs_s.log_softmax(dim=-1), outputs_t.softmax(dim=-1))
        loss_kd = criterion_list[2](outputs_s.log_softmax(dim=-1), outputs_t.softmax(dim=-1))

        # Combine losses (adjust weights as needed)
        if mode == "maximize":
            loss = -(loss_cls + loss_div + loss_kd)
        else:
            loss = loss_cls + loss_div + loss_kd

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs_s.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total

    return accuracy, epoch_loss  # Ensure these are floats, not tuples

def scrub(teacher, student, args, retain_loader_train, retain_loader_test, forget_loader_train, forget_loader_test, valid_loader_full):
    
    model_t = copy.deepcopy(teacher).eval()
    model_s = copy.deepcopy(student).eval()

    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = nn.KLDivLoss(reduction='batchmean')
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    if args.optim == "sgd":
        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay
        )
    elif args.optim == "adam": 
        optimizer = optim.Adam(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            weight_decay=args.sgda_weight_decay
        )
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(
            trainable_list.parameters(),
            lr=args.sgda_learning_rate,
            momentum=args.sgda_momentum,
            weight_decay=args.sgda_weight_decay
        )

    module_list.append(model_t)
    module_list.to(args.device)
    criterion_list.to(args.device)

    t1 = time.time()

    forget_validation_loader = forget_loader_test  # Use your provided forget test loader

    for epoch in range(1, args.sgda_epochs + 1):
        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        print("==> scrub unlearning ...")

        test_acc_r = calculate_accuracy(model_s, retain_loader_test, criterion_cls, args.device)
        test_acc_f = calculate_accuracy(model_s, forget_loader_test, criterion_cls, args.device)
        test_acc_v = calculate_accuracy(model_s, valid_loader_full, criterion_cls, args.device)
        test_acc_fv = calculate_accuracy(model_s, forget_validation_loader, criterion_cls, args.device)
        
        acc_r = test_acc_r['Acc']  
        acc_f = test_acc_f['Acc']
        acc_v = test_acc_v['Acc']
        acc_fv = test_acc_fv['Acc']

        maximize_acc, maximize_loss = train_distill(epoch, forget_loader_train, module_list, optimizer, criterion_list, args, "maximize")
        train_acc, train_loss = train_distill(epoch, retain_loader_train, module_list, optimizer, criterion_list, args, "minimize")

        print(f"Epoch {epoch}: maximize loss: {maximize_loss:.2f}, train_acc: {train_acc:.2f}, minimize loss: {train_loss:.2f}")

    t2 = time.time()
    print(f"Total time: {t2 - t1:.2f} seconds")

    return model_s  
