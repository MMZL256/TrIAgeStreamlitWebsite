import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import resnet34, ResNet34_Weights
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, pipeline
import kornia
from PIL import ImageOps
from streamlit_local_storage import LocalStorage
import time
from streamlit_back_camera_input import back_camera_input
st.markdown(
    """
    <style>
        [data-testid="stCameraInput"] {
            max-height: none !important;
            height: auto;
        }
        [data-testid="stCameraInput"] video {
            width: 100%;
            height: auto;
            max-height: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#LE MODÈLE

canny = kornia.filters.Canny(low_threshold=0.99, high_threshold=0.999)

v2Transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

kTransform = nn.Sequential(
    kornia.augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    kornia.color.RgbToGrayscale()
)

def transform(img):
    img = v2Transform(img)
    return img

#Class names
classes = ("BananaPeels", "CansLaterals", "CansTops", "JuiceBottlesLaterals",
        "JuiceBottlesTops", "JuiceBoxLaterals", "JuiceBoxTops", "MilkBoxes", 
        "SnackPackages", "UsedPaper")
frclasses = ("Pelure de banane", "Canette d'aluminium (côté)",
             "Canette d'aluminium (haut)", "Bouteille de jus (côté)",
             "Bouteille de jus (haut)", "Boîte de jus (côté)", 
             "Boîte de jus (haut)", "Boîte de lait", "Emballage", "Papier usé")

#Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet34 = resnet34(weights=None)
        # Replace the final fully connected layer
        in_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet34(x)
        return x

REPO_ID = "MMZL/TrIAge"
FILENAME = "wasteClassAD_RESNET34.pth"

@st.cache_resource(show_spinner="En train de charger le modèle...")
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    return state_dict
@st.cache_resource(show_spinner="En train d'accéder au modèle...")
def get_model():
    #Initialize model
    wasteClassModel = NeuralNetwork()
    pretrainedModelParams = load_model()
    wasteClassModel.load_state_dict(pretrainedModelParams)
    wasteClassModel.eval()
    return wasteClassModel

#LE SITE

st.set_page_config(page_title="CSL TrIAge", page_icon="♻️", layout="wide")

st.title("Opération TrIAge")
"Ceci est un site en développement. Modèle présentement utilisé: wasteClassAD"

localStorage = LocalStorage()
totalNum = localStorage.getItem("total") if localStorage.getItem("total") else 0
recycled = localStorage.getItem("recycling") if localStorage.getItem("recycling") else 0
composted = localStorage.getItem("compost") if localStorage.getItem("compost") else 0
trashed = localStorage.getItem("trash") if localStorage.getItem("trash") else 0
consigned = localStorage.getItem("consigning") if localStorage.getItem("consigning") else 0



enable = st.checkbox("Activer la caméra")
backCamera = st.checkbox("Utiliser la caméra arrière")
if backCamera:
    wastePicture_buffer = back_camera_input()
    st.write("Tapez l'écran pour prendre une photo")
else:
    wastePicture_buffer = st.camera_input(key="cameraInput", label="Prenez une photo de l'objet:", disabled=not enable)
output = None
if wastePicture_buffer is not None: 
    t0 = time.time()
    model = get_model()
    wasteImage = Image.open(wastePicture_buffer).convert("RGB")
    wasteImage = ImageOps.exif_transpose(wasteImage)
    wasteImage = transform(wasteImage)
    wasteImage = kTransform(wasteImage)
    _, wasteEdges = canny(wasteImage)
    wasteImage = wasteEdges.repeat(1, 3, 1, 1)
    shownTestImage = v2.functional.to_pil_image(wasteImage.squeeze(0))
    t1 = time.time()
    with torch.no_grad():
        output = model(wasteImage)
        output = torch.softmax(output, dim=1)
        confidence, index = torch.max(output, dim=1)
        t2 = time.time()
        predictedClass = classes[index.item()]
        frPredictedClass = frclasses[index.item()]
    localStorage.setItem("total", totalNum+1, key='set_total')
    if predictedClass in ["JuiceBoxLaterals", "JuiceBoxTops", "MilkBoxes", 
                            "SnackPackages"]:
        st.header(":blue-background[Recyclage]")
        #st.info(":blue-background[Recyclage]")
        st.write("Type d'objet détecté: " + frPredictedClass)
        if predictedClass in ["JuiceBoxLaterals", "JuiceBoxTops"]:
            st.write("La paille en papier va au compost!")
        localStorage.setItem("recycling", recycled+1, key='set_recycling')
    elif predictedClass in ["BananaPeels", "UsedPaper"]:
        st.header(":orange-background[Compost]")
        #st.info(":orange-background[Compost]")
        st.write("Type d'objet détecté: " + frPredictedClass)
        localStorage.setItem("compost", composted+1, key='set_compost')
    elif predictedClass in ["CansLaterals", "CansTops", "JuiceBottlesLaterals",
                            "JuiceBottlesTops"]:
        st.header(":green-background[Contenants consignés]")
        #st.info(":green-background[Contenants consignés]")
        st.write("Type d'objet détecté: " + frPredictedClass)
        localStorage.setItem("consigning", consigned+1, key='set_consigning')
    elif predictedClass in ["Shoes"]:
        st.header(":gray-background[Déchets]")
        #st.info(":gray-background[Déchets]")
        st.write("Type d'objet détecté: " + frPredictedClass)
        localStorage.setItem("trash", trashed+1, key="set_trash")
    with st.expander("Voir les détails"):
        "Image perçue par le modèle:"
        shownTestImage
        st.write("Type d'objet détecté: " + frPredictedClass)
        st.write(f"Certitude: {confidence.item()*100:.2f}%")
        st.write(f"Durée de prétraitement d'image: {t1-t0:.3f} secondes")
        st.write(f"Durée de d'inférence: {t2-t1:.3f} secondes")
with st.expander("Voir les statistiques"):
    "Voici les statistiques récoltés par cet appareil: "
    st.write(":rainbow-background[TOTAL] ", totalNum)
    st.write(":blue-background[Recyclage] ", recycled)
    st.write(":orange-background[Compost] ", composted)
    st.write(":green-background[Contenants consignés] ", consigned)
    st.write(":gray-background[Déchets] ", trashed)
    st.write(":rainbow-background[POINTAGE]", 10*composted + 10*consigned + 5*recycled + trashed)
    st.write("10 points par compostable, 10 points par contenant consigné, 5 points par recyclable, 1 point par déchet")

"Dernière mise à jour: 7 mars 2026"












