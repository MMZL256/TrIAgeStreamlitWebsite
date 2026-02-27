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
import streamlit_webrtc
import numpy as np
import av
import cv2
import time

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
        x = torch.flatten(x, start_dim=1)
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

model = get_model()
model.eval()
lastProcessTime = 0
processingInterval = 0.5
lastOutputArray = None
lastOutputFrame = None
def callback(frame):
    global lastProcessTime, lastOutputArray, lastOutputFrame, processingInterval
    currentTime = time.time()
    try:
        if currentTime - lastProcessTime >= processingInterval or lastOutputArray is None:
            lastProcessTime = currentTime

            PILimg = frame.to_image() # from VideoFrame to PIL
            edgeTensor = transform(PILimg) # Transformed to torch tensor chw

            modelInput = edgeTensor.unsqueeze(0) # bchw
            with torch.no_grad():
                output = model(modelInput)
                _, predictionIndex = torch.max(output, dim=1)
                label = classes[predictionIndex.item()]

            edgeTensor = edgeTensor[0]
            numpyEdges = (edgeTensor * 255).clamp(0, 255).byte().cpu().numpy() # To suitable np
            bgrEdges = np.repeat(numpyEdges[:, :, None], 3, axis=2) # to bgr
            bgrEdges = np.ascontiguousarray(bgrEdges)

            cv2.putText(
                bgrEdges,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
            lastOutputArray = bgrEdges
            lastOutputFrame = av.VideoFrame.from_ndarray(bgrEdges, format="bgr24")
        else:
            if lastOutputArray is not None:
                lastOutputFrame = av.VideoFrame.from_ndarray(lastOutputArray, format="bgr24")
    except Exception as e:
        print(f"Error in callback: {e}")
        if lastOutputArray is not None:
            lastOutputFrame = av.VideoFrame.from_ndarray(lastOutputArray, format="bgr24")
        elif lastOutputFrame is None:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            lastOutputFrame = av.VideoFrame.from_ndarray(blank, format="bgr24")
    return lastOutputFrame

streamlit_webrtc.webrtc_streamer(key="theStreamer", video_frame_callback=callback)
