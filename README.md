# 🖼️ Image Classifier

A web app that classifies images using deep learning — upload any image and AI identifies what's in it.

## 🔍 What it does
- Accepts any JPG or PNG image
- Returns top 5 predictions with confidence scores
- Highlights the most likely classification

## 🛠️ Built With
- Python
- PyTorch — deep learning framework
- ResNet50 — pretrained on ImageNet (1.2M images, 1000 categories)
- Streamlit — web interface

## ⚠️ Known Limitation
ResNet50 is limited to 1000 ImageNet categories. It may struggle with niche or uncommon objects not represented in its training data.

## 🚀 Run Locally
pip install streamlit torch torchvision pillow
python -m streamlit run app.py

## 👤 Author
Arsheyam — AI Student, 4th Semester