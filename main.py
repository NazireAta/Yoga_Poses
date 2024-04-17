import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os
from PIL import Image
import cv2
# import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow import keras
# from tensorflow.keras import layers


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.utils import save_image ## added
from torch.utils.data import Dataset ## added




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
         
def extract_frames(video_path, output_folder, fps=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    os.makedirs(output_folder, exist_ok=True)
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        if frame_index % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(output_path, frame)

        frame_index += 1
    cap.release()

def text_output(myarr):
    myfile = open("text_output.txt",'w')
    time = 0
    last = myarr[0]
    for pose in myarr:
        if pose==last:
            time+=1
        else:
            myfile.write("Hold {} for {} seconds\n".format(last,time))
            last = pose
            time = 1
    myfile.write("Hold {} for {} seconds\n".format(last,time))

if _name_ == '_main_':
    root_path = sys.path[0]

    input_video_path = os.path.join(root_path, 'input', 'myvideo.mp4')
    video_path = os.path.join(root_path, 'input', 'DATASET','VIDEO_TEST')
    output_folder = os.path.join(root_path, 'input', 'DATASET','VIDEO_TEST','none')
    desired_fps = 1

    extract_frames(input_video_path, output_folder, desired_fps)


    # Define data paths
    train_path = os.path.join(root_path, 'input', 'DATASET','TRAIN')
    validation_path = os.path.join(root_path, 'input', 'DATASET','VALIDATION')
    test_path = os.path.join(root_path, 'input', 'DATASET','TEST')
    positive_samples_path = os.path.join(root_path, 'input', 'DATASET', 'TRAIN') ## added
    negative_samples_path = os.path.join(root_path, 'NEGATIVE_SAMPLES') ## added

    os.makedirs(negative_samples_path, exist_ok=True) ## added

    if not (os.path.exists(train_path) or os.path.exists(validation_path) or os.path.exists(test_path)):
        print('input paths could not be resolved')

    # Define data transforms
    train_transforms = transforms.Compose([ #just introducing variations to the learning data
        transforms.RandomRotation(degrees=10), #random rotations upto 10 degrees may be applied
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.3), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Resize((224,224), antialias = True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #rescale to [0,1]
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224,224), antialias = True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    negative_transforms = transforms.Compose([ # added
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    ])

    for class_folder in os.listdir(positive_samples_path): ## added
        class_path = os.path.join(positive_samples_path, class_folder)
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)

            # Load the image
            img = loader(image_path)

            # Apply transformations
            transformed_img = negative_transforms(img)

            # Save the transformed image to the negative samples folder
            save_path = os.path.join(negative_samples_path, f"{class_folder}_{image_file}")
            save_image(transformed_img, save_path)


    class CustomDataset(Dataset): ## added
        def _init_(self, root, transform=None, loader=None):
            self.positive_dataset = ImageFolder(root=os.path.join(root, 'DATASET', 'TRAIN'), transform=transform, loader=loader)
            self.negative_dataset = ImageFolder(root=os.path.join(root, 'NEGATIVE_SAMPLES'), transform=transform, loader=loader)

        def _getitem_(self, index):
            if index < len(self.positive_dataset):
                return self.positive_dataset[index]
            else:
                # Adjust the index for negative samples
                index -= len(self.positive_dataset)
                return self.negative_dataset[index]

        def _len_(self):
            return len(self.positive_dataset) + len(self.negative_dataset)


    #HYPERPARAMETERS
    batch_size = 8
    num_workers = 4


    train_dataset = CustomDataset(root=root_path, transform=train_transforms, loader=loader) ## added
    #train_dataset = ImageFolder(root=train_path, transform=train_transforms, loader=loader)
    validation_dataset = ImageFolder(root=validation_path, transform=test_transforms, loader=loader)
    test_dataset = ImageFolder(root=test_path, transform=test_transforms, loader=loader)

    video_dataset = ImageFolder(root = video_path, transform=test_transforms, loader=loader)

    #Define the loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) # added
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    video_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Define the encoder
    encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
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

    train_losses = []
    val_losses = []

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        for batch_num, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = encoder(inputs)
            outputs = classifier(outputs.view(outputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss+=loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        epoch_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = encoder(inputs)
                outputs = classifier(outputs.view(outputs.size(0), -1))
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_validation_loss = epoch_val_loss / len(validation_loader)
        val_losses.append(avg_validation_loss)
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}")
    
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.show()

    # Testing and evaluation
    correct = 0
    total = 0
    class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = encoder(inputs)
            outputs = classifier(outputs.view(outputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")


    class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
    predictions = []
    with torch.no_grad():
        for inputs, _ in video_loader:
            outputs = encoder(inputs)
            outputs = classifier(outputs.view(outputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend([class_names[idx] for idx in predicted.cpu().numpy()])

    df = pd.DataFrame({"Predictions": predictions})
    df.to_excel('predictions.xlsx', index=False) 
    text_output(predictions)
