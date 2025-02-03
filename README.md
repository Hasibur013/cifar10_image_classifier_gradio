# 🎯 CIFAR-10 Image Classification with Gradio, PyTorch & MLflow

## 📌 Project Overview
This project builds and deploys an image classification model using the **CIFAR-10 dataset**. The model is trained using **PyTorch Lightning**, tracked with **MLflow**, and deployed as a web app using **Gradio**.

## 🚀 Features
- **Exploratory Data Analysis (EDA)** with `pandas` and `seaborn`
- **Custom Data Pipeline** using `torchvision.transforms`
- **Model Training** using `PyTorch Lightning`
- **Experiment Tracking** with `MLflow`
- **Deployment** using `Gradio`
- **Pretrained Model (ResNet-18) for High Accuracy(85) using simple it was 79**
- **Supports Image Upload in PNG, JPG Formats**

## 🛠️ Tools & Technologies Used
- **Python** 🐍
- **PyTorch & PyTorch Lightning** 🔥
- **MLflow** 🏷️
- **Gradio** 🌐
- **NumPy & Pandas** 📊
- **Seaborn & Matplotlib** 📉

## 📂 Project Structure
```
📁 image_classifier
│-- 📂 data
│-- 📂 models
│-- 📂 lightning_logs
│-- 📂 mlruns
│-- app.py   # Gradio Web App
│-- model_build.ipynb  # Model Training Script
│-- requirements.txt
│-- README.md
```

## 📸 Model Deployment (Gradio Web App)
Run the following command to launch the web app:
```bash
python app.py
```
The app allows you to upload an image and get predictions from the trained model.

## 🎥 Project Demo
[![Project Demo](https://img.youtube.com/vi/afcKd1glXXg/0.jpg)](https://youtu.be/afcKd1glXXg?si=yhQ0b6ASNy26TZAW)

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Hasibur013/cifar10_image_classifier_gradio.git
   ```
2. Navigate to the project folder:
   ```bash
   cd image-classifier
   ```
3. Create virtual Environment:
   ```bash
   Python -m venv venv
   ```
4. Active virtual Environment:
   ```bash
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Train the Model
Run the training script:
```bash
notebook model_build.ipynb
```

## 📌 Run MLflow Tracking UI
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser to visualize training metrics.

## 🌟 Future Enhancements
- [ ] Support additional image datasets
- [ ] Improve model accuracy with hyperparameter tuning
- [ ] Deploy as a cloud-based service

## 🚀 Run Web App
Run the app script:
```bash
python app.py
```
Upload an image (PNG, JPG, etc.) to classify it into one of the CIFAR-10 categories. Image will be airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

---
Made with ❤️ by Md. Hasibur Rahman

