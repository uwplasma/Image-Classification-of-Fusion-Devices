# Stellarator Image Classification with Machine Learning

## Project Overview
This project utilizes machine learning to classify stellarator fusion devices based on image analysis. The primary objective is to identify the number of field periods.

## Key Features
- **Classify periodicity**: Automatically determine the number of field periods from images.
- **Deep learning model**: Uses a Convolutional Neural Network based on ResNet-18 for feature extraction.
- **Custom dataset**: Trained on labeled images of stellarators to improve prediction accuracy. The dataset was generated using the code inside the `pyQSC` folder and contains labeled images with corresponding field period values.

## Model Development
The model was trained on a dataset of stellarator images labeled with their field periods. Images were preprocessed by resizing, normalizing, and converting them into tensors before being fed into a modified ResNet-18 model.

ResNet-18 is a deep learning architecture designed for image recognition, using residual connections to improve training efficiency. We adapted it for regression by modifying its final layer to output a single continuous value.

### Training Process
- The dataset was split into training, validation, and test sets.
- The ResNet-18 model's fully connected layer was modified for regression.
- Training used Mean Squared Error (MSE) loss and the Adam optimizer.
- Validation was performed after each epoch to track performance.

## Inference and Prediction
- The trained model predicts field periodicity from unseen images.
- Preprocessing ensures consistency across inputs.

## Using the Pre-Saved Model for Predictions

There are two ways to use the model for predictions:  

### 1. Predicting a Specific Image  
You can run predictions on any image included in the project or use your own image. Follow these steps:  

1. Open the `prediction_images` folder.  
2. Inside, you will find two subdirectories:  
   - `online_images/` → Contains images added manually or from external sources.  
   - `script_images/` → Contains images generated from the `pyQSC` code.  
3. Find the image you want to predict and note its filename.  
4. Open the `predict_image.py` file.  
5. Locate the image name in the script and replace it with the filename you want to predict.  
6. Ensure you are using the correct model for prediction. Model files can be found under `model_implementation/saved_models/`.  
7. Run the script to get the predicted field period value.  
8. If you want to add your own images, place them in `prediction_images/online_images/` for organization.  

#### Naming Scheme for Prediction Images
Images inside `prediction_images/` follow this naming format: `field_{nfp}_v{version}.png`
 - `nfp`: Represents the number of field periods of the fusion device in the image.
 - `version`: Indicates the image version or sequence number.

### 2. Predicting All Images in the Folder  
If you want to run predictions on all images inside `prediction_images/`, follow these steps:  

1. Run the `predict_all_images.py` script.  
2. The script will automatically process all images in both `online_images/` and `script_images/`.  
3. The results will be displayed, showing actual vs. predicted values for each image.  


## Training a New Model  

You can also train a new model for predictions by following these steps:  

1. **Install all Dependencies**
   - Run `./install.sh` to create a virtual envirnonemnt and install all dependencies that are needed

2. **Reactivate Virtual Environment (When Returning to the Project)**  
   - Once you've run `./install.sh`, you don’t need to run it again.  
   - Whenever you return to work on this project, activate the virtual environment using:  
      
      ```bash
      source venv/bin/activate
      ```
3. **Generate Image Data**  
   - Run `data_collection_script.py` located in the `scripts/` folder.  
   - This will create an image folder named `stel_images/`.  

4. **Create a Dataset**  
   - Run `dataset_script.py` located in the `scripts/` folder, which will process the `stel_images/` folder.  
   - This script generates a dataset containing image paths and their corresponding field periods.  
   - The generated dataset will be saved in the `model_implementation/datasets/` folder.  

5. **Train the Model**  
   - Run `train_and_save_model.py` inside the `model_implementation/` folder.  
   - Update the dataset path in the script to point to the newly created dataset.  
   - Change the model save path to store the new model in `model_implementation/saved_models/`.  

6. **Use the New Model for Predictions**  
   - Once training is complete, follow the steps in the [Using the Pre-Saved Model for Predictions](#using-the-pre-saved-model-for-predictions) section to predict images using the newly trained model.  

### Note  
If you want details on model implementation, training, or inference processes, or wish to modify them, refer to the following files inside the `model_implementation/` folder:  

- `plasma_dataset.py`  
- `plasma_inference.py`  
- `plasma_model.py`  
- `plasma_train.py`  

## Online Image Sources  

The online images used in this project were obtained from the following resources:  

- [APS Physics - Quasisymmetric Stellarators](https://physics.aps.org/articles/v15/5)  
- [ResearchGate](https://www.researchgate.net/figure/D-representation-of-the-W7-X-coils-shapes-in-orange-and-the-separatrix-in-blue-From_fig3_354253660)  
- [Wikipedia - Stellarator](https://en.wikipedia.org/wiki/Stellarator)  

## Acknowledgements
- [pyQSC](https://github.com/landreman/pyQSC) for generating images.
- Pedro Curvo, Diogo R. Ferreira, and R. Jorge for providing the dataset of plottable stellarators. [Dataset link](https://zenodo.org/records/13623959).
