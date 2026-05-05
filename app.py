import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #080808;
        color: #f0f0f0;
    }

    .header {
        text-align: center;
        padding: 2.5rem 0 1rem;
    }
    .header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff, #888888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header p {
        color: #555;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .divider {
        border: none;
        border-top: 0.5px solid #222;
        margin: 1.5rem 0;
    }

    .top-prediction {
        background: linear-gradient(135deg, #1a1a1a, #111);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .top-prediction .label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-transform: capitalize;
    }
    .top-prediction .confidence {
        font-size: 0.9rem;
        color: #00e5a0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    .pred-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 0.5px solid #1a1a1a;
        font-size: 0.9rem;
    }
    .pred-row .name { color: #ccc; text-transform: capitalize; }
    .pred-row .pct { color: #555; font-variant-numeric: tabular-nums; }

    .bar-wrap {
        background: #1a1a1a;
        border-radius: 99px;
        height: 4px;
        width: 100%;
        margin: 0.2rem 0 0.6rem;
    }
    .bar-fill {
        background: #00e5a0;
        height: 4px;
        border-radius: 99px;
    }

    .model-badge {
        display: inline-block;
        background: #111;
        border: 0.5px solid #2a2a2a;
        border-radius: 99px;
        padding: 0.4rem 1rem;
        font-size: 0.75rem;
        color: #555;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 1.5rem;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header">
        <h1>Image Classifier</h1>
        <p>ResNet50 · PyTorch · ImageNet · by Arsheyam</p>
    </div>
    <hr class="divider">
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        labels = json.load(f)
    return labels

with st.spinner("Loading model..."):
    model = load_model()
    labels = load_labels()

uploaded_file = st.file_uploader("Drop an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, width=500)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probabilities, 5)

    top_label = labels[top5.indices[0].item()].replace("_", " ")
    top_prob = top5.values[0].item() * 100

    st.markdown(f"""
        <div class="top-prediction">
            <div class="label">{top_label}</div>
            <div class="confidence">Confidence: {top_prob:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Other possibilities**")
    for i in range(1, 5):
        label = labels[top5.indices[i].item()].replace("_", " ")
        prob = top5.values[i].item() * 100
        st.markdown(f"""
            <div class="pred-row">
                <span class="name">{label}</span>
                <span class="pct">{prob:.1f}%</span>
            </div>
            <div class="bar-wrap"><div class="bar-fill" style="width:{prob:.1f}%"></div></div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center"><span class="model-badge">ResNet50 trained on ImageNet · 1000 categories</span></div>', unsafe_allow_html=True)