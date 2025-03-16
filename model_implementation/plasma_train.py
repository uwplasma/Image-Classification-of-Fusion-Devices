import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from plasma_dataset import PlasmaDataset
from plasma_model import PlasmaCNN

def train_model(csv_path, transform, model_save_path, batch_size=32, num_epochs=11, lr=1e-4):
    """
    Trains the PlasmaCNN model on the given dataset.

    Args:
        csv_path (str): Path to the CSV file containing image paths and labels.
        transform (callable): Transformation to apply to images.
        model_save_path (str): Path where the trained model will be saved.
        batch_size (int, optional): Number of samples per batch (default: 32).
        num_epochs (int, optional): Number of training epochs (default: 11).
        lr (float, optional): Learning rate for the optimizer (default: 1e-4).
    
    Returns:
        tuple: (trained model, test data loader)
    """
    
    # Load the dataset
    plasma_dataset = PlasmaDataset(csv_file=csv_path, transform=transform)
    
    # Split the dataset: 80% train, 10% validation, 10% test
    train_indices, temp_indices = train_test_split(range(len(plasma_dataset)), test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Create dataset subsets
    train_dataset = Subset(plasma_dataset, train_indices)
    val_dataset = Subset(plasma_dataset, val_indices)
    test_dataset = Subset(plasma_dataset, test_indices)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = PlasmaCNN()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Compute average validation loss
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print("Model saved!")
    
    return model, test_loader

def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test dataset.
    
    Args:
        model (nn.Module): The trained PlasmaCNN model.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            
            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    # Compute average test loss
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")