# 🎧 SyncPixel 📸  
*Bringing Music to Your Mood from an Image*

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DeepFace](https://img.shields.io/badge/Face%20Analysis-DeepFace-blue)](https://github.com/serengil/deepface)
[![HuggingFace](https://img.shields.io/badge/Image%20Captioning-BLIP-yellow)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🔍 Overview

**SyncPixel** is an AI-powered music recommendation system that analyzes an image to:
- Generate a **caption** (context understanding),
- Detect **emotions** via facial analysis,
- Recommend **YouTube music videos** that align with the mood and content of the image.

---

## ✨ Features

- 🎭 Emotion Detection with DeepFace
- 🧠 Image Captioning via BLIP (HuggingFace Transformers)
- 🔍 Smart YouTube Search Query Generation
- 🎵 Language & Genre Filters (English/Hindi, Pop, Hip-hop, etc.)
- 📽️ Embedded YouTube Music Player
- 📊 Emotion Bar Charts with Matplotlib

---

## 🖼️ Demo

> Coming Soon — [Demo Video](#)  
*(Optional: Add a Streamlit Share or YouTube demo link here.)*

---

## ⚙️ Tech Stack

| Category          | Technology                                   |
|------------------|-----------------------------------------------|
| Web Framework     | [Streamlit](https://streamlit.io/)            |
| Face Analysis     | [DeepFace](https://github.com/serengil/deepface) |
| Captioning Model  | [BLIP - Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Video API         | [YouTube Data API v3](https://developers.google.com/youtube/v3) |
| Visuals & Charts  | Matplotlib, PIL, OpenCV                      |
| Others            | Regex, Transformers, Google API Client       |

---

## 🚀 Getting Started

### 🔑 Prerequisites

- Python 3.7+
- Valid YouTube Data API key ([Get one here](https://console.cloud.google.com/))
- Hugging Face access (for model downloading)

### 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/syncpixel.git
cd syncpixel

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
