import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, pipeline
import kornia
from PIL import ImageOps
from streamlit_local_storage import LocalStorage

st.markdown(
    """
    <style>
        /* Target the camera widget container */
        [data-testid="stCameraInput"] {
            max-height: none !important;   /* Remove any default max-height */
            height: auto;                   /* Let height be determined by content */
        }
        /* Make the video element fill the width and scale height automatically */
        [data-testid="stCameraInput"] video {
            width: 100%;
            height: auto;
            max-height: none;               /* Override any video max-height */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#LE MODÈLE

#Image normalizer transform
canny = kornia.filters.Canny(low_threshold=0.2, high_threshold=0.25)
v2Transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
])

def transform(img):
    img = v2Transform(img)
    imgbchw = img.unsqueeze(0)
    _, imgedges = canny(imgbchw)
    #shownEdges = kornia.tensor_to_image(imgedges.byte())
    return imgedges.squeeze(0)

#Class names
#classes = ("AluminumFoil", "BananaPeel", "Bottles", "Cans", "Cardboard", "Cups", "FoodWaste",
#           "Gobelets", "GobeletsLids", "JuiceBottles", "JuiceBoxes", "MilkBoxes", "Plastic", 
#           "PlasticBags", "Shoes", "SnackPackages", "Straws", "Styrofoam", "UsedBrownPaper",
#           "UsedWhitePaper")

classes = ("AluminumFoil", "CansLaterals", "CansTops", 
           "Gobelets", "GobeletsLids", "JuiceBottlesLaterals", "JuiceBottlesTops",
           "JuiceBoxLaterals", "JuiceBoxTops", "MilkBoxes", "SandwichPackage", "SnackPackages",
           "UsedPaper", "WaterBottlesLaterals")
frclasses = ("Papier aluminium", "Canette d'aluminium (côté)", "Canette d'aluminium (haut)", 
             "Gobelet", "Couvercle de gobelet", "Bouteille de jus (côté)", "Bouteille de jus (haut)",
             "Boîte de jus (côté)", "Boîte de jus (haut)", "Carton de lait", "Boîte à sandwich", 
             "Emballage de plastique métallisé", "Papier usé", "Bouteille d'eau de plastique (côté)")

#Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolutional stage
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        #Full connections (y=Wx+b)
        self.fc1 = nn.Linear(179776, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 14)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=0, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

REPO_ID = "MMZL/TrIAge"
FILENAME = "wasteClassTEST14.pth"

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
"Ceci est un site en développement. Modèle présentement utilisé: wasteClassTEST14"

localStorage = LocalStorage()
totalNum = localStorage.getItem("total") if localStorage.getItem("total") else 0
recycled = localStorage.getItem("recycling") if localStorage.getItem("recycling") else 0
composted = localStorage.getItem("compost") if localStorage.getItem("compost") else 0
trashed = localStorage.getItem("trash") if localStorage.getItem("trash") else 0
consigned = localStorage.getItem("consigning") if localStorage.getItem("consigning") else 0

with st.expander("Voir les statistiques"):
    typeCol, numCol = st.columns(spec=2, gap=None, width=500)
    with typeCol:
        st.write(":rainbow-background[TOTAL] ")
        st.write(":blue-background[Recyclage] ")
        st.write(":orange-background[Compost] ")
        st.write(":green-background[Contenants consignés] ")
        st.write(":gray-background[Déchets] ")
    with numCol:
        st.write(totalNum)
        st.write(recycled)
        st.write(composted)
        st.write(consigned)
        st.write(trashed)

enable = st.checkbox("Activer la caméra")
wastePicture_buffer = st.camera_input(key="cameraInput", label="Prenez une photo de l'objet:", disabled=not enable)
output = None
if wastePicture_buffer is not None: 
    model = get_model()
    wasteImage = Image.open(wastePicture_buffer).convert("RGB")
    wasteImage = ImageOps.exif_transpose(wasteImage)
    wasteImage = transform(wasteImage)
    shownTestImage = v2.functional.to_pil_image(wasteImage)
    shownTestImage
    with torch.no_grad():
        output = model(wasteImage)
        _, index = torch.max(output, dim=0)
        predictedClass = classes[index.item()]
        frPredictedClass = frclasses[index.item()]
        st.write("Type d'objet détecté: " + frPredictedClass)
        localStorage.setItem("total", totalNum+1, key='set_total')
        if predictedClass in ["AluminumFoil", "Gobelets", "GobeletsLids", 
                                "JuiceBoxLaterals", "JuiceBoxTops", "MilkBoxes", 
                                "SandwichPackage", "SnackPackages"]:
            st.header(":blue-background[Recyclage]")
            localStorage.setItem("recycling", recycled+1, key='set_recycling')
        if predictedClass in ["UsedPaper"]:
            st.header(":orange-background[Compost]")
            localStorage.setItem("compost", composted+1, key='set_compost')
        if predictedClass in ["WaterBottlesLaterals", "CansTops", "CansLaterals",
                                "JuiceBottlesLaterals", "JuiceBottlesTops"]:
            st.header(":green-background[Contenants consignés]")
            localStorage.setItem("consigning", consigned+1, key='set_consigning')
        if predictedClass in ["Shoes"]:
            st.header(":gray-background[Déchets]")
            localStorage.setItem("trash", trashed+1, key="set_trash")
