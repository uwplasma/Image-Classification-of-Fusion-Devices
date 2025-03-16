from plasma_train import train_model, evaluate_model
from plasma_inference import predict
from torchvision import transforms

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Specify dataset and model save paths (update as needed)
csv_path = "./datasets/color_dataset.csv"  # Path to the dataset CSV file
model_save_path = "./saved_models/test1.pth"  # Path to save the trained model

# Train the model and return the trained model along with the test data loader
model, test_loader = train_model(csv_path, transform, model_save_path)

# Evaluate the trained model on the test dataset
evaluate_model(model, test_loader)

print("Done training.")