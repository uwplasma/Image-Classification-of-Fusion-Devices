import os
import re
import torch
from plasma_inference import predict
from plasma_model import PlasmaCNN
from torchvision import transforms
import warnings

warnings.simplefilter("ignore", UserWarning)

# Set device to GPU if available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

model = PlasmaCNN()

# Load the trained model weights
model.load_state_dict(torch.load("./saved_models/plasma_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Directories to search for images
directories = ["../prediction_images/online_images", "../prediction_images/script_images"]

# Regular expression pattern to extract actual field periods from filenames
pattern = re.compile(r"field_(\d+)_v\d+\.png")

# Tracking accuracy
total_images = 0
correct_predictions = 0

results = []

for directory in directories:
    if not os.path.exists(directory):
        continue  

    files = [f for f in os.listdir(directory) if pattern.match(f)]
    
    if not files:
        continue  

    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue  

        # Extract actual field period
        actual_value = int(match.group(1))  
        image_path = os.path.join(directory, filename)

        with torch.no_grad():
            predicted_value = round(predict(image_path, model, transform))  

        # Store result including folder name
        folder_name = os.path.basename(directory)
        results.append((folder_name, filename, actual_value, predicted_value))

        # Accuracy tracking
        total_images += 1
        if actual_value == predicted_value:
            correct_predictions += 1

# Print results
if results:
    print("Processing images...\n")
    print(f"{'Folder':<15} {'Image Name':<25} {'Actual Value':<15} {'Predicted Value':<15}")
    print("=" * 75)
    for folder, filename, actual, predicted in results:
        print(f"{folder:<15} {filename:<25} {actual:<15} {predicted:<15}")

    # Compute accuracy
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print("\nAccuracy: {:.2f}%".format(accuracy))
else:
    print("No valid images found in the specified directories.")
