import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

#-------------------------------------------------------
#LOAD AND PREPROCESS DATA

# Custom Dataset Class
class PlasmaDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image path and label
        image_path = self.data.iloc[idx, 0]
        label = float(self.data.iloc[idx, 1])  # Assume labels are float
        
        # Open and process the image
        image = Image.open(image_path).convert("RGB")  # Ensure images are RGB
        if self.transform:
            image = self.transform(image)
        
        # Return the processed image and label as tensors
        return image, torch.tensor(label, dtype=torch.float32)

# Define Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Path to dataset.csv
csv_path = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/model_implementation/dataset.csv"

# Initialize the Dataset
plasma_dataset = PlasmaDataset(csv_file=csv_path, transform=transform)

# Create a DataLoader
data_loader = DataLoader(plasma_dataset, batch_size=32, shuffle=True)

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

# Split the dataset into 80% for training and 20% for validation + test
train_indices, temp_indices = train_test_split(range(len(plasma_dataset)), test_size=0.2, random_state=42)

# Now split the 20% temp_indices into 50% for validation and 50% for testing
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# Create Subsets for training, validation, and testing
train_dataset = Subset(plasma_dataset, train_indices)
val_dataset = Subset(plasma_dataset, val_indices)
test_dataset = Subset(plasma_dataset, test_indices)

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#----------------------------------------------------

#cnn model
import torch.nn as nn
import torchvision.models as models

# Define the CNN Model
class PlasmaCNN(nn.Module):
    def __init__(self):
        super(PlasmaCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Load  
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1),  # Modify for regression
            nn.ReLU()  # Ensure positive outputs
        )
    
    def forward(self, x):
        return self.base_model(x)

# Initialize model
model = PlasmaCNN()

#----------------------------------------------------
#  Define Loss Function and Optimizer
import torch.optim as optim

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#------------------------------------------------------
# Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
# num_epochs = 11

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         # Forward pass
#         outputs = model(images).squeeze()
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
    
#     train_loss /= len(train_loader)
    
#     # Validation loop
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
            
#             outputs = model(images).squeeze()
#             loss = criterion(outputs, labels)
            
#             val_loss += loss.item()
    
#     val_loss /= len(val_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


#-------------------------------------------------------
#save and evaluate model
# Save the model
# torch.save(model.state_dict(), "plasma_cnn.pth")
# print("Model saved!")

# Load the model (for inference later)
model.load_state_dict(torch.load("plasma_cnn.pth"))
model.eval()


#------------------------------------------------
#perform inference

#Testing loop
test_loss = 0.0
with torch.no_grad():  # Disable gradient computation for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images).squeeze()
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()

# Average test loss
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")



# Single image inference
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).item()
    return output

# Example
image_path = "/home/exouser/Public/Image-Classification-of-Fusion-Devices-/model_implementation/test_image112246_2.png"
predicted_field_periods = predict(image_path, model)
print(f"Predicted field periods: {predicted_field_periods:.2f}")
