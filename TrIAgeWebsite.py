import streamlit as st
import pandas as pd

st.title("Opération TrIAge")
st.write("Hello World!!! :)")

enable = st.checkbox("Enable camera")
wastePicture = st.camera_input("Prenez une photo de l'objet:",
                             disabled=not enable)

if wastePicture:
    st.image(wastePicture)