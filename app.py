import streamlit as st
import numpy as np
import time

st.set_page_config(page_title ="🌾 Angkorice Vision", layout="wide")

st.markdown("""
<div style="text-align:center; margin-top:40px;">
    <h1 style="font-size:48px; margin-bottom:5px;">🌾 Angkorice Vision</h1>
</div>
""", unsafe_allow_html=True)


st.subheader("Rice Leaf Disease Detection and Advisory System", text_alignment="center")


# Upload Botton
uploaded = st.file_uploader("Upload your rice leaf images 🍚🍃 :", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    try:
        with st.spinner():
            time.sleep(3)
        st.success("Sucessfuly uploaded image!")
    except Exception as e:
        st.error("Error occured during image")
    
    if st.button("Predict"):
        tr
