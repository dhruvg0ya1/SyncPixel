"""Vision models: BLIP captioning/VQA and DeepFace emotion classification."""

import streamlit as st
import torch
from PIL import Image
from deepface import DeepFace
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import re



@st.cache_resource(show_spinner="Loading BLIP model…")
def _blip():
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return proc, model

def blip_caption(img: Image.Image) -> str:
    proc, model = _blip()
    inputs = proc(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, num_beams=5, early_stopping=True)
    return proc.decode(out[0], skip_special_tokens=True)

def blip_vqa(img: Image.Image, question: str) -> str:
    proc, model = _blip()
    inputs = proc(img, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, num_beams=5, early_stopping=True)
    return proc.decode(out[0], skip_special_tokens=True)

def analyse_image_blip(img: Image.Image) -> dict:
    general   = blip_caption(img)
    happening = blip_vqa(img, "what is happening in this image")
    who_what  = blip_vqa(img, "who or what is in this image")
    setting   = blip_vqa(img, "describe the setting and environment")
    objects   = blip_vqa(img, "list the main objects in this image")

    description = (
        f"📷 <b>Scene:</b> {general}<br>"
        f"🎬 <b>Happening:</b> {happening}<br>"
        f"👤 <b>Subject:</b> {who_what}<br>"
        f"🏞️ <b>Setting:</b> {setting}<br>"
        f"📦 <b>Objects:</b> {objects}"
    )

    all_text = f"{general} {happening} {who_what} {setting} {objects}".lower()
    stop = {"a","the","is","are","in","on","with","and","of","to","at","this","that",
            "an","image","photo","picture","there","some","their","its","was","being"}
    keywords = list({w for w in re.findall(r'\b[a-z]{3,}\b', all_text) if w not in stop})[:18]

    return {
        "description": description,
        "description_plain": f"{general}. {happening}. {who_what}. {setting}. {objects}",
        "general": general,
        "keywords": keywords,
    }

def analyse_emotion(img: Image.Image) -> dict:
    try:
        res = DeepFace.analyze(
            img_path=np.array(img), actions=["emotion"],
            enforce_detection=False, silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        return dict(dominant=res["dominant_emotion"], scores=res["emotion"], face=True)
    except:
        return dict(dominant="neutral", scores={"neutral": 100.0}, face=False)
