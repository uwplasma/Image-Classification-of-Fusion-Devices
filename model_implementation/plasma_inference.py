import torch
from PIL import Image
from plasma_model import PlasmaCNN
from torchvision import transforms

def predict(image_path, model, transform):
    """
    Perform inference on a single image using the trained model.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): The trained PlasmaCNN model.
        transform (callable): The transformation pipeline applied to the image.

    Returns:
        float: The model's predicted output value for the image.
    """

    # Load the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")

    # Apply the specified transformations and add batch dimension
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Perform inference without computing gradients
    with torch.no_grad():
        output = model(image).item()

    return output
