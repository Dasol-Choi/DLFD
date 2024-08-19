import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(data_dir, batch_size=64, val_split=0.1, retain_split=0.7, seed=42):
    
    random.seed(seed)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0)),  # Random cropping
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=None) 

    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)

    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_remaining = num_train - num_val  # train - val
    num_retain = int(num_remaining * retain_split)
    num_forget = num_remaining - num_retain

    val_indices = indices[:num_val]
    retain_indices = indices[num_val:num_val + num_retain]
    forget_indices = indices[num_val + num_retain:]

    val_dataset = Subset(full_train_dataset, val_indices)
    retain_dataset = Subset(full_train_dataset, retain_indices)
    forget_dataset = Subset(full_train_dataset, forget_indices)

    train_indices = retain_indices + forget_indices
    train_dataset = Subset(full_train_dataset, train_indices)

    val_dataset.dataset.transform = test_transform
    train_dataset.dataset.transform = train_transform
    retain_dataset.dataset.transform = train_transform
    forget_dataset.dataset.transform = train_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    retain_train_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    forget_train_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)

    retain_test_dataset = Subset(full_train_dataset, retain_indices)
    forget_test_dataset = Subset(full_train_dataset, forget_indices)

    retain_test_dataset.dataset.transform = test_transform
    forget_test_dataset.dataset.transform = test_transform

    retain_test_loader = DataLoader(retain_test_dataset, batch_size=batch_size, shuffle=False)
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_loaders = {
        'train': train_loader,
        'retain_train': retain_train_loader,
        'forget_train': forget_train_loader,
        'val': val_loader,
        'test': test_loader,
        'retain_test': retain_test_loader,
        'forget_test': forget_test_loader
    }

    print(f"Train dataset size: {len(train_dataset)}")  # retain + forget
    print(f"Train (Retain) dataset size: {len(retain_dataset)}")
    print(f"Forget (Train) dataset size: {len(forget_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Retain Test dataset size: {len(retain_test_dataset)}")
    print(f"Forget Test dataset size: {len(forget_test_dataset)}")

    return data_loaders
