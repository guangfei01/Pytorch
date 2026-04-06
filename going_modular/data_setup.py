# setup train and test dataloaders
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0

def create_dataloaders(train_dir, test_dir, batch_size, transform, num_workers=NUM_WORKERS):
    # create training and test data loaders
    # Takes in a training and test directory and turns them into PyTorch dataloaders
    # with transforms and target transforms applied
    # Args:
    #   train_dir: directory containing training data
    #   test_dir: directory containing test data
    #   batch_size: batch size for dataloaders
    #   num_workers: number of workers for dataloaders (default: number of CPU cores)
    # Returns:
    #   train_dataloader: dataloader for training data
    #   test_dataloader: dataloader for test data
    # Define transforms for training and test datata

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    
    return train_dataloader, test_dataloader, class_names

    #


