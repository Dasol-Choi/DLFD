import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

class CustomDataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None, forget=False, retain=False):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        image_age_list = parsing(meta_data)
        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        full_path = os.path.join(self.image_directory, image_path)
        img = Image.open(full_path).convert('RGB')  # Ensure the image is in RGB format
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label

def get_data_loaders(train_meta_data_path, train_image_directory,
                     val_meta_data_path, val_image_directory,
                     test_meta_data_path, test_image_directory, batch_size=64):
    
    # Load metadata
    train_meta_data = pd.read_csv(train_meta_data_path)
    val_meta_data = pd.read_csv(val_meta_data_path)  # This will become 'test'
    test_meta_data = pd.read_csv(test_meta_data_path)  # This will become 'unseen'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CustomDataset(train_meta_data, train_image_directory, train_transform)
    val_dataset = CustomDataset(val_meta_data, val_image_directory, transform)
    test_dataset = CustomDataset(test_meta_data, test_image_directory, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Renaming 'val' to 'test'
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Renaming 'test' to 'unseen'

    # Create forget and retain data loaders for train and test
    forget_dataset_train = CustomDataset(train_meta_data, train_image_directory, train_transform, forget=True)
    forget_loader_train = DataLoader(forget_dataset_train, batch_size=batch_size, shuffle=True)

    retain_dataset_train = CustomDataset(train_meta_data, train_image_directory, train_transform, retain=True)
    retain_loader_train = DataLoader(retain_dataset_train, batch_size=batch_size, shuffle=True)

    forget_dataset_test = CustomDataset(train_meta_data, train_image_directory, transform, forget=True)
    forget_loader_test = DataLoader(forget_dataset_test, batch_size=batch_size, shuffle=False)

    retain_dataset_test = CustomDataset(train_meta_data, train_image_directory, transform, retain=True)
    retain_loader_test = DataLoader(retain_dataset_test, batch_size=batch_size, shuffle=False)

    # Construct the data loaders dictionary
    data_loaders = {
        'train': train_loader,
        'test': test_loader,  
        'val': val_loader,  
        'retain_train': retain_loader_train,
        'retain_test': retain_loader_test,
        'forget_train': forget_loader_train,
        'forget_test': forget_loader_test
    }

    return data_loaders