from plasma_inference import predict
from torchvision import transforms
from plasma_model import PlasmaCNN
import torch
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

# Initialize the PlasmaCNN model
model = PlasmaCNN()

# Load the trained model weights (Change Model Name if Needed)
model.load_state_dict(torch.load("./saved_models/plasma_cnn.pth", map_location=device))

# Move the model to the selected device (CPU/GPU)
model.to(device)

# Set the model to evaluation mode 
model.eval()                      

# Path to the image for inference (Change Image name)
image_path = "../prediction_images/online_images/field_4_v1.png"

with torch.no_grad():
    predicted_field_periods = predict(image_path, model, transform)
# Round output to the nearest integer 
print(f"Predicted field periods: {round(predicted_field_periods)}")