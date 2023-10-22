import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os
from PIL import Image
# import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow import keras
# from tensorflow.keras import layers
# from torchvision.prototype.models import resnet50, ResNet50_Weights


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models


def loader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if os.path.getsize(path) < 5:
                    print(f"Image file is truncated or too small: {path}")
                if img is None:
                    print(f"Image file could not be loaded: {path}")
                else:
                    return img.convert("RGB")
    except Exception as e:
        print(f"Error loading image: {path}. Exception: {str(e)}")
         


if __name__ == '__main__':
    root_path = sys.path[0]

    # Define data paths

    train_path = os.path.join(root_path, 'input', 'DATASET','TRAIN')
    validation_path = os.path.join(root_path, 'input', 'DATASET','VALIDATION')
    test_path = os.path.join(root_path, 'input', 'DATASET','TEST')
    if not (os.path.exists(train_path) or os.path.exists(validation_path) or os.path.exists(test_path)):
        print('input paths could not be resolved')

    # Define data transforms
    train_transforms = transforms.Compose([ #just introducing variations to the learning data
        transforms.RandomRotation(degrees=10), #random rotations upto 10 degrees may be applied
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.3), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #rescale to [0,1]
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #HYPERPARAMETERS
    # Create data loaders
    batch_size = 8
    num_workers = 4

    train_dataset = ImageFolder(root=train_path, transform=train_transforms, loader=loader)
    validation_dataset = ImageFolder(root=validation_path, transform=test_transforms, loader=loader)
    test_dataset = ImageFolder(root=test_path, transform=test_transforms, loader=loader)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Define the encoder
    encoder = models.resnet50(True)
    modules = list(encoder.children())[:-1]
    encoder = nn.Sequential(*modules)

    # Define a classifier
    num_classes = len(train_dataset.classes)
    classifier = nn.Sequential(
        nn.Linear(2048, num_classes),  # Modify 2048 to match the output of your encoder
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = encoder(inputs)
            outputs = classifier(outputs.view(outputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation loop (optional)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = encoder(inputs)
                outputs = classifier(outputs.view(outputs.size(0), -1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")

    # Testing and evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = encoder(inputs)
            outputs = classifier(outputs.view(outputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")