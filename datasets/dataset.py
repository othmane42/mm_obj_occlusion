"""Loading dataset useful functions."""

import torch
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder #, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
import collections
from torchvision import transforms

def load_label_file(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        parts = line.split()
        label = parts[0]
        values = [float(part) for part in parts[1:]]
        labels.append((label, values))

    return labels

class_to_index = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}
# Define your dataset class and data loaders
class CustomDataset(Dataset):
    def __init__(self, image_folder, voxel_folder, label_folder):
        self.image_folder = image_folder
        self.voxel_folder = voxel_folder
        self.label_folder = label_folder
        self.image_paths = sorted(os.listdir(image_folder))
        self.voxel_paths = sorted(os.listdir(voxel_folder))
        self.label_paths = sorted(os.listdir(label_folder))
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to a consistent size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image data
        image = Image.open(os.path.join(self.image_folder, self.image_paths[idx]))  # Load and preprocess image data
        if self.image_transform:
            image = self.image_transform(image)


        # Load and preprocess voxel data
        voxel_data = np.load(os.path.join(self.voxel_folder, self.voxel_paths[idx]))
        target_shape = (32, 32, 32)
        pad_value = 0
        padded_voxel_data = np.pad(voxel_data, [(0, max(target_shape[0] - voxel_data.shape[0], 0)),
                                                (0, max(target_shape[1] - voxel_data.shape[1], 0)),
                                                (0, max(target_shape[2] - voxel_data.shape[2], 0))], mode='constant', constant_values=pad_value)
        cropped_voxel_data = padded_voxel_data[:target_shape[0], :target_shape[1], :target_shape[2]]
        voxel_data_resized = torch.from_numpy(cropped_voxel_data).float()

        # Load and preprocess label data
        label_lines = open(os.path.join(self.label_folder, self.label_paths[idx])).readlines()
        target = {}
        #print("hello 1")
        target['boxes'] = []
        target['labels'] = []

        for line in label_lines:
            parts = line.strip().split()
            if parts[0] in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']:
                target['labels'].append(class_to_index[parts[0]])  # Convert class to index and add to labels
                target['boxes'].append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

        # Pad the target['boxes'] tensor to the same length
        max_num_boxes = max(len(target['boxes']), 10)#10  # Adjust this based on your maximum expected number of boxes

        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)  # Convert labels to tensor
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        voxel_data_resized = voxel_data_resized.unsqueeze(dim=0)
        return image, voxel_data_resized, target
    
def collate_fn_map(batch):
    images, voxel_data, targets = zip(*batch)

    # Determine the maximum voxel size within the batch
    max_voxel_shape = max(data.shape for data in voxel_data)

    # Pad or crop voxel_data tensors to the same size within the batch
    padded_voxel_data = torch.stack(
        [F.pad(data, (0, max_voxel_shape[2] - data.shape[2], 0, max_voxel_shape[1] - data.shape[1], 0, max_voxel_shape[0] - data.shape[0]))
         if data.shape != max_voxel_shape
         else data
         for data in voxel_data]
    )
    print("padded voxel data in fn shape is ",padded_voxel_data.shape)
    return torch.stack(images), padded_voxel_data, targets
              

def data_loader(image_folder,voxel_folder,label_folder,batch_size=3):
    """Takes input dataset and colnames and generates
    a PyTorch DataLoader."""
    # Create data loaders
    train_dataset = CustomDataset(image_folder, voxel_folder, label_folder)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_map)  
    return train_data_loader
    

  