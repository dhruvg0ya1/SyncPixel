"""Gemini text generation: captions and hashtags."""

import streamlit as st
import re
import google.generativeai as genai


GEMINI_API_KEY       = st.secrets["GEMINI_API_KEY"]


@st.cache_resource
def _gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")

def gemini_text(prompt: str) -> str:
    try:
        return _gemini().generate_content([prompt]).text.strip()
    except Exception as e:
        return f"[error: {e}]"

def gen_captions(emotion: str, plain_desc: str, exclude: list, n: int = 3) -> list[str]:
    excl = ("\n\nNEVER reuse any of these:\n" + "\n".join(f"- {c}" for c in exclude)) if exclude else ""
    prompt = f"""Write exactly {n} viral social media captions.

Scene: {plain_desc[:500]}
Emotion/Mood: {emotion}
{excl}

Rules:
• Output ONLY bullet lines, each starting with •
• Include relevant emojis in each caption
• Mix tones: 1 inspirational, 1 witty/playful, 1 emotionally resonant
• Max 160 characters each
• No headers, numbers, or explanations — just the bullet list"""
    raw = gemini_text(prompt)
    caps, seen = [], {c.lower() for c in exclude}
    for line in raw.splitlines():
        c = line.lstrip("•-*· ").strip()
        if c and c.lower() not in seen and len(c) > 10:
            caps.append(c)
            seen.add(c.lower())
    return caps[:n]

def gen_hashtags(emotion: str, plain_desc: str, exclude: list, n: int = 5) -> list[str]:
    excl = (f"\nNEVER use any of: {' '.join(exclude)}") if exclude else ""
    prompt = f"""Generate exactly {n} trending viral hashtags for a social media post.

Emotion: {emotion}
Scene: {plain_desc[:400]}
{excl}

Output: ONE line only, {n} hashtags separated by spaces, format: #tag1 #tag2 #tag3
No explanations, no extra lines, no bullet points."""
    raw = gemini_text(prompt)
    tags, seen, result = re.findall(r"#\w+", raw), {e.lower() for e in exclude}, []
    for tag in tags:
        if tag.lower() not in seen:
            result.append(tag)
            seen.add(tag.lower())
    return result[:n]
