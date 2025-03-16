import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class PlasmaDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading plasma images and their corresponding labels.
    
    Attributes:
        data (DataFrame): DataFrame containing image paths and labels.
        transform (callable, optional): Transformation function to preprocess images.
    """
    
    def __init__(self, csv_file, transform=None):
        """
        Initializes the dataset by reading image paths and labels from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data = pd.read_csv(csv_file)  # Load CSV file into a Pandas DataFrame
        self.transform = transform  # Store image transformation function

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, label) where:
                - image (Tensor): Preprocessed image tensor.
                - label (Tensor): Corresponding label as a float tensor.
        """
        # Extract image path and label from the DataFrame
        image_path = self.data.iloc[idx, 0]  # First column contains image paths
        label = float(self.data.iloc[idx, 1])  # Second column contains labels (convert to float)
        
        # Open the image and ensure it's in RGB format
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Return image and label as tensors
        return image, torch.tensor(label, dtype=torch.float32)
