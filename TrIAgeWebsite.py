import streamlit as st
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms

#Image normalizer transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Opération TrIAge")
st.write("Hello World!!! :)")

enable = st.checkbox("Activer la caméra")
wastePicture_buffer = st.camera_input("Prenez une photo de l'objet:",
                             disabled=not enable)

if wastePicture_buffer is not None:
    wasteImage = transform(wastePicture_buffer)
