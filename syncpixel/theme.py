"""Colour themes driven by the detected emotion."""

import streamlit as st


THEMES = {
    "happy":    dict(bg="#0d0b00", primary="#FFD700", secondary="#8a6800",
                     card="#1a1600", border="#FFD700", accent="#FFE97A",
                     glow="rgba(255,215,0,0.12)"),
    "sad":      dict(bg="#00070d", primary="#4A9EF5", secondary="#1a3570",
                     card="#000f1f", border="#4A9EF5", accent="#93C5FD",
                     glow="rgba(74,158,245,0.12)"),
    "angry":    dict(bg="#0e0000", primary="#FF4444", secondary="#7a0000",
                     card="#1c0000", border="#FF4444", accent="#FCA5A5",
                     glow="rgba(255,68,68,0.12)"),
    "fear":     dict(bg="#06000f", primary="#9B6CF7", secondary="#3a1270",
                     card="#0f0020", border="#9B6CF7", accent="#C4B5FD",
                     glow="rgba(155,108,247,0.12)"),
    "disgust":  dict(bg="#000f00", primary="#3DCC6E", secondary="#0b4019",
                     card="#001a00", border="#3DCC6E", accent="#86EFAC",
                     glow="rgba(61,204,110,0.12)"),
    "surprise": dict(bg="#0c0500", primary="#FF8C00", secondary="#7a2c00",
                     card="#170a00", border="#FF8C00", accent="#FED7AA",
                     glow="rgba(255,140,0,0.12)"),
}


def get_theme(emotion: str):
    return THEMES.get(emotion.lower() if emotion else "")

def inject_theme(emotion: str):
    t = get_theme(emotion)
    if not t:
        return
    st.markdown(f"""<style>
.stApp {{background-color:{t['bg']};}}
section[data-testid="stSidebar"] {{background-color:{t['card']};border-right:1px solid {t['border']}40;}}
h1,h2,h3,h4 {{color:{t['primary']};}}
.stButton>button {{background:{t['secondary']};color:#fff;border:1px solid {t['border']};border-radius:8px;font-weight:600;transition:all .2s;}}
.stButton>button:hover {{background:{t['primary']};color:#000;border-color:{t['primary']};}}
.stProgress>div>div>div {{background:{t['primary']};}}
.stRadio>div,label {{color:#e0e0e0 !important;}}
.sp-card {{background:{t['card']};border:1px solid {t['border']}55;border-radius:14px;padding:18px;margin:10px 0;transition:all .25s;}}
.sp-card:hover {{border-color:{t['border']};box-shadow:0 0 24px {t['glow']};}}
.caption-box {{background:{t['card']};border-left:4px solid {t['primary']};border-radius:8px;padding:12px 16px;margin:8px 0;color:#f0f0f0;font-size:15px;}}
.hashtag-pill {{display:inline-block;background:{t['secondary']};color:{t['accent']};border:1px solid {t['border']};border-radius:20px;padding:5px 14px;margin:4px;font-size:14px;font-weight:600;letter-spacing:.4px;}}
.analysis-box {{background:{t['card']};border:1px solid {t['border']}55;border-radius:12px;padding:16px;max-height:460px;overflow-y:auto;color:#ddd;font-size:14px;line-height:1.8;white-space:pre-wrap;}}
.emotion-badge {{background:{t['secondary']};border:2px solid {t['border']};border-radius:12px;padding:14px;text-align:center;margin-bottom:12px;}}
.genre-chip {{display:inline-block;background:{t['card']};color:{t['primary']};border:1px solid {t['border']}55;border-radius:16px;padding:3px 10px;margin:3px;font-size:12px;}}
</style>""", unsafe_allow_html=True)
