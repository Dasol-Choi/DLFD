import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from evaluation.accuracy import calculate_accuracy  
from evaluation.mia import MIA  

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def linear_weight_scheduler(current_batch, max_batches, initial_weight, final_weight):
    weight_increment = (final_weight - initial_weight) / (max_batches - 1)
    current_weight = initial_weight + weight_increment * current_batch
    return current_weight

def compute_ot_loss(logits_retain, logits_forget, epsilon=0.01):
    C = get_cost_matrix(logits_retain, logits_forget)
    T = sinkhorn(C, epsilon, niter=50)
    ot_loss = torch.sum(T * C)
    return ot_loss

def get_cost_matrix(x_feature, y_feature):
    C_fea = cost_matrix_cos(x_feature, y_feature)
    return C_fea

def cost_matrix_cos(x, y, temperature=1.0, p=2):
    x_col = x.unsqueeze(1) / temperature
    y_lin = y.unsqueeze(0) / temperature

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    c = torch.clamp(1 - cos(x_col, y_lin), min=0)
    return c

def sinkhorn(C, epsilon, niter=50, device='cuda'):
    m = C.size(0)
    n = C.size(1)
    mu = Variable(1. / m * torch.FloatTensor(m).fill_(1).to(device), requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1).to(device), requires_grad=False)

    tau = -.8
    thresh = 10**(-1)

    def M(u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)

    u, v, err = torch.zeros_like(mu), torch.zeros_like(nu), 0.
    for i in range(niter):
        u1 = u
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        err = (u - u1).abs().sum()

        if err.item() < thresh:
            break

    U, V = u, v
    pi = torch.exp(M(U, V))
    pi = pi.to(device).float()
    return pi

def train_dlfd(dlfm_model, data_loaders, criterion, optimizer, device, perturbation_steps, perturbation_strength, forget_threshold, max_forgetting_score, initial_retain_weight, final_retain_weight, forget_weight, path):

    print("Starting Forgetting Phase...")
    dataloader_iterator = iter(data_loaders['forget_test'])

    max_batches = len(data_loaders['retain_test'])

    best_test_acc = 0  # Initialize best test accuracy
    forgetting_score = 3  # Initialize forgetting_score

    for batch_idx, (x_retain, y_retain) in enumerate(data_loaders['retain_test']):
        if batch_idx >= max_batches:
            break

        y_retain = y_retain.cuda()

        try:
            (x_forget, y_forget) = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(data_loaders['forget_test'])
            (x_forget, y_forget) = next(dataloader_iterator)

        if x_forget.size(0) != x_retain.size(0):
            continue

        retain_weight = linear_weight_scheduler(batch_idx, max_batches, initial_retain_weight, final_retain_weight)

        if forgetting_score >= forget_threshold:
            noise = torch.zeros_like(x_retain, requires_grad=True)
        
            for i in range(perturbation_steps):
                perturbed_retain = x_retain + noise
                perturbed_retain = torch.clamp(perturbed_retain, 0, 1).cuda()
                perturbed_feature = dlfm_model(perturbed_retain.cuda())

                forget_feature = dlfm_model(x_forget.cuda())
                outputs_retain = dlfm_model(x_retain.cuda())

                # Compute OT loss and classification loss
                ot_loss = compute_ot_loss(outputs_retain, forget_feature) * forget_weight
                cls_loss = criterion(perturbed_feature, y_retain) * retain_weight
                total_loss = -cls_loss + ot_loss

                optimizer.zero_grad()
                total_loss.backward()
                noise.data += perturbation_strength * torch.sign(noise.grad.data)
                noise.grad.zero_()

            # Train the model with the perturbed image
            perturbed_retain = x_retain + noise
            perturbed_retain = torch.clamp(perturbed_retain, 0, 1).detach().cuda()

            outputs = dlfm_model(perturbed_retain.cuda())
            loss = criterion(outputs, y_retain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print(f'Forgetting score below threshold. Fine-tuning with only classification loss.')
            
            outputs = dlfm_model(x_retain.cuda())
            loss = criterion(outputs, y_retain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = calculate_accuracy(dlfm_model, data_loaders['val'], criterion, device)
        mia = MIA(dlfm_model, data_loaders["retain_test"], data_loaders['forget_test'], data_loaders['val'], device)
        forgetting_score = mia["Forgetting Score"]
        final_score = (test_acc["Acc"] + 1 - mia["Forgetting Score"] * 2) / 2 

        print(f'Batch {batch_idx + 1}/{len(data_loaders["retain_test"])} - Test Acc: {test_acc["Acc"]:.4f}, MIA Forgetting Score: {forgetting_score:.4f}, Final Score: {final_score:.4f}')

        if forgetting_score <= max_forgetting_score:
            if test_acc["Acc"] > best_test_acc:
                best_test_acc = test_acc["Acc"]
                torch.save(dlfm_model.state_dict(), path)
                print(f"New best model saved with Test Acc: {test_acc['Acc']:.4f}")
