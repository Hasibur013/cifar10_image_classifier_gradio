# 🎯 CIFAR-10 Image Classification with PyTorch & MLflow

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
│-- 📂 notebooks
│-- 📂 artifacts
│-- app.py   # Gradio Web App
│-- train.py  # Model Training Script
│-- mlflow_tracking.py  # MLflow Experiment Logging
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
[![Project Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-classifier.git
   ```
2. Navigate to the project folder:
   ```bash
   cd image-classifier
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Train the Model
Run the training script:
```bash
python train.py
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

---
Made with ❤️ by Your Name

