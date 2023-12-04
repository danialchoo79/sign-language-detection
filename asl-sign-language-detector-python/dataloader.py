import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


class BoundingBoxFeatures(Dataset):
    def __init__(self, root='sliced_sequences', split='train'):
        super(BoundingBoxFeatures, self).__init__()
        self.root = f'{root}/{split}'
        data_dict = {}
        for folder in os.listdir(self.root):
            for filename in os.listdir(f'{self.root}/{folder}'):
                data_dict[filename] = folder
        self.data = list(data_dict.items())
        self.label_map = {'j': 0, 'o': 1, 'z': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, label = self.data[index]
        X = pd.read_csv(f'{self.root}/{label}/{filename}').to_numpy().astype('float32')
        X = X[:, :2]
        y = self.label_map[label]
        return X, y
    

class TrailImageDataset(Dataset):
    def __init__(self, root):
        super(TrailImageDataset, self).__init__()
        self.root = root
        data_dict = {}
        for folder in os.listdir(root):
            for filename in os.listdir(f'{root}/{folder}'):
                data_dict[filename] = folder
        self.data = list(data_dict.items())
        self.label_map = {'j': 0, 'o': 1, 'z': 2}
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, label = self.data[index]
        image_path = f'{self.root}/{label}/{filename}'
        
        # Open the image using PIL and convert to B&W
        image = Image.open(image_path).convert('L').transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert image to a torch tensor and normalize
        X = self.transform(image)
        
        # Map label to its corresponding index
        y = torch.tensor(self.label_map[label])
        
        return X, y


if __name__ == "__main__":
    dataset = BoundingBoxFeatures('sliced_sequences', 'test')
    X, y = dataset.__getitem__(3)
    print(X)
    print(X.shape, y)
