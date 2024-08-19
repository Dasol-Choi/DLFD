import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evaluation.accuracy import calculate_accuracy

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, student_output, teacher_output):
        student_log_prob = torch.log_softmax(student_output / self.T, dim=1)
        teacher_prob = torch.softmax(teacher_output / self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(student_log_prob, teacher_prob) * (self.T * self.T)
        return loss

def adjust_learning_rate(optimizer, epoch, lr_decay_epochs, lr_decay_rate, learning_rate):
    lr = learning_rate * (lr_decay_rate ** sum(epoch >= np.array(lr_decay_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_bad_teacher(epoch, retain_loader, forget_loader, module_list, criterion_list, optimizer, args):
    model_s = module_list[0]  # Student model
    model_gt = module_list[1] # Good teacher model
    model_bt = module_list[2] # Bad teacher model

    model_s.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(retain_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        optimizer.zero_grad()

        outputs_s = model_s(inputs)

        with torch.no_grad():
            outputs_gt = model_gt(inputs)

        with torch.no_grad():
            outputs_bt = model_bt(inputs)

        loss_cls = criterion_list[0](outputs_s, labels) # Classification Loss
        loss_div = criterion_list[1](outputs_s, outputs_gt) # Good Teacher KD Loss
        loss_kd = criterion_list[2](outputs_s, outputs_bt) # Bad Teacher KD Loss
        loss = loss_cls + args.bt_alpha * loss_div - args.bt_beta * loss_kd

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs_s.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(retain_loader.dataset)
    accuracy = 100. * correct / total

    return accuracy, epoch_loss

def badt(gteacher, bteacher, student, retain_loader, forget_loader, valid_loader_full, args):

    model_gt = copy.deepcopy(gteacher)
    model_bt = copy.deepcopy(bteacher)
    model_s = copy.deepcopy(student)

    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.bt_kd_T)
    criterion_kd = DistillKL(args.bt_kd_T)

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    if args.bt_optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              momentum=args.bt_momentum,
                              weight_decay=args.bt_weight_decay)
    elif args.bt_optim == "adam": 
        optimizer = optim.Adam(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              weight_decay=args.bt_weight_decay)
    elif args.bt_optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                              lr=args.bt_learning_rate,
                              momentum=args.bt_momentum,
                              weight_decay=args.bt_weight_decay)

    module_list.append(model_gt)
    module_list.append(model_bt)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    acc_rs = []
    acc_fs = []
    acc_vs = []
    
    print("==> Bad Teacher Unlearning ...")
    for epoch in range(1, args.bt_epochs + 1):
        # Validate on each dataset
        acc_r = calculate_accuracy(model_s, retain_loader, criterion_cls, args.device)['Acc']
        acc_f = calculate_accuracy(model_s, forget_loader, criterion_cls, args.device)['Acc']
        acc_v = calculate_accuracy(model_s, valid_loader_full, criterion_cls, args.device)['Acc']

        acc_rs.append(100 - acc_r)
        acc_fs.append(100 - acc_f)
        acc_vs.append(100 - acc_v)

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.bt_lr_decay_epochs, args.bt_lr_decay_rate, args.bt_learning_rate)

        # Train the student model
        train_acc, train_loss = train_bad_teacher(epoch, retain_loader, forget_loader, module_list, criterion_list, optimizer, args)

        print(f"Epoch {epoch}: train_acc: {train_acc:.2f}, train_loss: {train_loss:.2f}")

    return model_s
