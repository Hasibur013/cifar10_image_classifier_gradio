import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import mlflow.pytorch
import numpy as np

# Define CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model from MLflow
tracking_uri = "F:\\bongoDev ML Course\\000.Exercises\\image_classifier\\mlruns"
experiment_id = "408912435176137303"
run_id = "a4a4fb9883c74f08ac04c8d8cb00cf35"
model_uri = f"{tracking_uri}\\{experiment_id}\\{run_id}\\artifacts\\model"

# Load model using MLflow
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict_image(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0].numpy()
        prediction_idx = np.argmax(probabilities)
        predicted_class = CLASSES[prediction_idx]
        confidence = probabilities[prediction_idx] * 100  # Convert to percentage
    return {predicted_class: confidence}


# Launch the app
if __name__ == '__main__':
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="CIFAR-10 Image Classifier App",
        description="Upload an image (PNG, JPG, etc.) to classify it into one of the CIFAR-10 categories. (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)"
    )
    interface.launch(share=True)
