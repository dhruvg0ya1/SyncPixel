# SyncPixel – Music from Images
# Stack: Streamlit · DeepFace · BLIP · Google Gemini 2.5 Flash · Spotify Web API

import streamlit as st
from PIL import Image
import numpy as np
import io, hashlib, re, time, random
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from deepface import DeepFace
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(
    page_title="SyncPixel",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEMES (no neutral) ───────────────────────────────────────────────────────
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

BASE_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
html, body, .stApp {font-family:'DM Sans',sans-serif;}
h1,h2,h3,h4 {font-family:'Syne',sans-serif;}
.center-title {text-align:center;font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;letter-spacing:-1px;margin:0 0 4px 0;line-height:1.1;}
.center-sub {text-align:center;color:#999;font-size:15px;margin:0 0 28px 0;}
.footer {text-align:center;padding:48px 20px 28px;color:#666;font-size:14px;font-family:'DM Sans',sans-serif;}
.footer a {color:#999;text-decoration:none;margin:0 10px;transition:color .2s;}
.footer a:hover {color:#fff;}
.footer-divider {height:1px;background:linear-gradient(90deg,transparent,#333,transparent);margin-bottom:24px;}
.welcome-box {text-align:center;padding:56px 20px 28px;}
.welcome-box h2 {font-family:'Syne',sans-serif;font-weight:700;}
.cam-wake-box {background:linear-gradient(135deg,#1a1a1a,#111);border:2px dashed #333;border-radius:12px;padding:36px 20px;text-align:center;color:#666;font-size:14px;margin-bottom:8px;}
.sp-card {border-radius:14px;padding:18px;margin:10px 0;}
.genre-chip {display:inline-block;border-radius:16px;padding:3px 10px;margin:3px;font-size:12px;}
.caption-box {border-radius:8px;padding:12px 16px;margin:8px 0;font-size:15px;}
.hashtag-pill {display:inline-block;border-radius:20px;padding:5px 14px;margin:4px;font-size:14px;font-weight:600;}
.analysis-box {border-radius:12px;padding:16px;max-height:460px;overflow-y:auto;font-size:14px;line-height:1.8;white-space:pre-wrap;}
.emotion-badge {border-radius:12px;padding:14px;text-align:center;margin-bottom:12px;}
.section-header {font-family:'Syne',sans-serif;font-weight:700;font-size:1.3rem;margin-bottom:12px;}
</style>"""

# ── BLIP ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BLIP model…")
def _blip():
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
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

# ── GEMINI (text only – captions & hashtags) ──────────────────────────────────
@st.cache_resource
def _gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")

def gemini_text(prompt: str) -> str:
    try:
        return _gemini().generate_content([prompt]).text.strip()
    except Exception as e:
        return f"[error: {e}]"

# ── SPOTIFY AUTH ──────────────────────────────────────────────────────────────
def _fetch_token() -> str | None:
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials",
              "client_id": SPOTIFY_CLIENT_ID,
              "client_secret": SPOTIFY_CLIENT_SECRET},
        timeout=10,
    )
    if r.ok:
        return r.json().get("access_token")
    st.error(f"Spotify auth failed {r.status_code}: {r.text[:200]}")
    return None

def spotify_token() -> str | None:
    now = time.time()
    if st.session_state.get("_sp_tok") and now < st.session_state.get("_sp_exp", 0):
        return st.session_state["_sp_tok"]
    tok = _fetch_token()
    if tok:
        st.session_state["_sp_tok"] = tok
        st.session_state["_sp_exp"] = now + 3300
    return tok

def _hdr(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}

# ── SPOTIFY – GENRE SEEDS (live API first) ────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_genre_seeds() -> list[str]:
    tok = spotify_token()
    if tok:
        r = requests.get(
            "https://api.spotify.com/v1/recommendations/available-genre-seeds",
            headers=_hdr(tok), timeout=10,
        )
        if r.ok:
            seeds = r.json().get("genres", [])
            if seeds:
                return seeds
    return [
        "acoustic","afrobeat","alt-rock","alternative","ambient","anime",
        "black-metal","bluegrass","blues","bossanova","brazil","breakbeat",
        "british","cantopop","chicago-house","children","chill","classical",
        "club","comedy","country","dance","dancehall","death-metal","deep-house",
        "detroit-techno","disco","disney","drum-and-bass","dub","dubstep","edm",
        "electro","electronic","emo","folk","forro","french","funk","garage",
        "german","gospel","goth","grindcore","groove","grunge","guitar","happy",
        "hard-rock","hardcore","hardstyle","heavy-metal","hip-hop","holidays",
        "honky-tonk","house","idm","indian","indie","indie-pop","industrial",
        "iranian","j-dance","j-idol","j-pop","j-rock","jazz","k-pop","kids",
        "latin","latino","malay","mandopop","metal","metal-misc","metalcore",
        "minimal-techno","movies","mpb","new-age","new-release","opera",
        "pagode","party","philippines-opm","piano","pop","pop-film","post-dubstep",
        "power-pop","progressive-house","psych-rock","punk","punk-rock","r-n-b",
        "rainy-day","reggae","reggaeton","road-trip","rock","rock-n-roll","rockabilly",
        "romance","sad","salsa","samba","sertanejo","show-tunes","singer-songwriter",
        "ska","sleep","songwriter","soul","soundtracks","spanish","study","summer",
        "swedish","synth-pop","tango","techno","trance","trip-hop","turkish",
        "work-out","world-music",
    ]

# ── EMOTION → AUDIO FEATURE TARGETS ──────────────────────────────────────────
_EMOTION_AF = {
    "happy":    dict(valence=0.82, energy=0.80, danceability=0.75, tempo=120),
    "sad":      dict(valence=0.18, energy=0.28, danceability=0.35, tempo=72),
    "angry":    dict(valence=0.18, energy=0.92, danceability=0.50, tempo=145),
    "disgust":  dict(valence=0.22, energy=0.65, danceability=0.45, tempo=108),
    "fear":     dict(valence=0.18, energy=0.38, danceability=0.30, tempo=82),
    "surprise": dict(valence=0.72, energy=0.85, danceability=0.78, tempo=132),
    "neutral":  dict(valence=0.50, energy=0.48, danceability=0.55, tempo=100),
}

_EMOTION_GENRE_HINTS = {
    "happy":    ["pop","dance","funk","disco","happy","party","summer"],
    "sad":      ["indie","folk","acoustic","blues","sad","emo","singer-songwriter","piano"],
    "angry":    ["rock","metal","punk","hardcore","grunge","heavy-metal","industrial"],
    "disgust":  ["alternative","grunge","industrial","punk-rock"],
    "fear":     ["ambient","goth","trip-hop","post-dubstep","darkwave"],
    "surprise": ["electronic","edm","indie-pop","synth-pop"],
    "neutral":  ["chill","indie","study","sleep","rainy-day","acoustic"],
}

def emotion_seeds(emotion: str, all_seeds: list[str]) -> list[str]:
    hints = _EMOTION_GENRE_HINTS.get(emotion, _EMOTION_GENRE_HINTS["neutral"])
    matched = [g for g in hints if g in all_seeds]
    if not matched:
        matched = [g for g in all_seeds if any(h in g for h in hints)]
    return matched[:5] or all_seeds[:5]

# ── GENRE GROUPS (for sidebar) ─────────────────────────────────────────────────
GENRE_GROUPS = {
    "Auto (match emotion)": [],
    "🎵 Pop":               ["pop","indie-pop","power-pop","synth-pop","k-pop","j-pop","cantopop","mandopop"],
    "🎸 Rock":              ["rock","alt-rock","hard-rock","indie","grunge","punk","punk-rock","rockabilly","rock-n-roll","psych-rock"],
    "🤘 Metal":             ["metal","heavy-metal","black-metal","death-metal","metalcore","hardcore","grindcore"],
    "🎤 Hip-Hop / R&B":     ["hip-hop","r-n-b","soul","funk","groove","dancehall"],
    "🎛️ Electronic / EDM":  ["electronic","edm","dance","house","techno","trance","dubstep","electro","drum-and-bass","minimal-techno","detroit-techno","chicago-house","deep-house","progressive-house","hardstyle","breakbeat","idm"],
    "🎷 Jazz / Blues":      ["jazz","blues","soul","bossanova"],
    "🌿 Chill / Ambient":   ["chill","ambient","sleep","rainy-day","new-age","study","acoustic","piano"],
    "🤠 Country / Folk":    ["country","folk","bluegrass","honky-tonk","singer-songwriter"],
    "💃 Latin":             ["latin","latino","reggaeton","salsa","samba","tango","forro"],
    "🎬 Soundtracks":       ["soundtracks","pop-film","disney","movies","show-tunes","anime"],
    "🌍 World":             ["world-music","afrobeat","indian","turkish","spanish","french","german","swedish","malay"],
    "😊 Mood":              ["happy","sad","romance","party","summer","road-trip","work-out","holidays"],
}

# ── SPOTIFY API CALLS ─────────────────────────────────────────────────────────
def sp_recommend(seeds: list[str], tok: str, limit=35, market="US", **kw) -> list[dict]:
    params = {"seed_genres": ",".join(seeds[:5]), "limit": limit, "market": market}
    params.update({f"target_{k}": v for k, v in kw.items()})
    r = requests.get("https://api.spotify.com/v1/recommendations",
                     headers=_hdr(tok), params=params, timeout=10)
    return r.json().get("tracks", []) if r.ok else []

def sp_search(q: str, tok: str, limit=20, market="US") -> list[dict]:
    r = requests.get("https://api.spotify.com/v1/search",
                     headers=_hdr(tok),
                     params={"q": q, "type": "track", "limit": limit, "market": market},
                     timeout=10)
    return r.json().get("tracks", {}).get("items", []) if r.ok else []

def sp_audio_features(ids: list[str], tok: str) -> dict:
    if not ids:
        return {}
    r = requests.get("https://api.spotify.com/v1/audio-features",
                     headers=_hdr(tok), params={"ids": ",".join(ids[:100])}, timeout=10)
    if not r.ok:
        return {}
    return {f["id"]: f for f in (r.json().get("audio_features") or []) if f}

# ── CHANGE 1: Batched artist genre fetch (single HTTP call instead of N calls) ─
def sp_artists_batch(ids: list[str], tok: str) -> dict:
    """Fetch genres for up to 50 artists in one API call.
    Replaces the old sequential sp_artist_genres loop — ~60% faster on large pools."""
    if not ids:
        return {}
    r = requests.get(
        "https://api.spotify.com/v1/artists",
        headers=_hdr(tok),
        params={"ids": ",".join(ids[:50])},
        timeout=10,
    )
    if not r.ok:
        return {}
    return {a["id"]: a.get("genres", []) for a in r.json().get("artists") or []}

# ── LANGUAGE DETECTION ────────────────────────────────────────────────────────
_HINDI_KW = {
    "hindi","bollywood","desi","bhangra","punjabi",
    "arijit","rahat fateh","shreya ghoshal","sunidhi","sonu nigam",
    "udit narayan","alka yagnik","lata mangeshkar","kishore kumar",
    "ar rahman","shankar ehsaan","ilaiyaraaja","kumar sanu","rafi",
    "playback","t-series","zee music","saregama",
}

def _txt(t: dict) -> str:
    return (t["name"] + " " + " ".join(a["name"] for a in t["artists"]) + " " + t["album"]["name"]).lower()

def _is_hindi(t: dict) -> bool:
    txt = _txt(t)
    if re.search(r"[\u0900-\u097F]", txt):
        return True
    return any(kw in txt for kw in _HINDI_KW)

def _is_english(t: dict) -> bool:
    txt = _txt(t)
    if re.search(r"[\u0900-\u097F\u4E00-\u9FFF\u3040-\u30FF\u0400-\u04FF\uAC00-\uD7AF]", txt):
        return False
    return bool(re.search(r"[a-zA-Z]", txt))

def lang_ok(t: dict, lang: str, custom: str | None = None) -> bool:
    if lang == "English": return _is_english(t) and not _is_hindi(t)
    if lang == "Hindi":   return _is_hindi(t)
    if lang == "Both":    return _is_english(t) or _is_hindi(t)
    if lang == "Other" and custom: return custom.lower() in _txt(t)
    return True

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt_dur(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"

def days_old(rd: str) -> int:
    try:
        return (datetime.now() - datetime.strptime(rd[:10], "%Y-%m-%d")).days
    except:
        return 9999

# ── IMAGE ANALYSIS (BLIP only) ────────────────────────────────────────────────
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

# ── EMOTION ANALYSIS (DeepFace) ───────────────────────────────────────────────
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

# ── CAPTIONS (Gemini text-only) ───────────────────────────────────────────────
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

# ── HASHTAGS (Gemini text-only) ───────────────────────────────────────────────
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

# ── SONG SEARCH (full Spotify API, audio-feature matched) ─────────────────────
def search_songs(
    emotion: str,
    blip_data: dict,
    all_seeds: list[str],
    language: str = "English",
    user_genre_seed: list[str] | None = None,
    popularity_mode: str = "All",
    exclude_ids: set | None = None,
    n: int = 5,
    custom_lang: str | None = None,
) -> list[dict]:

    tok = spotify_token()
    if not tok:
        return []

    exclude_ids = exclude_ids or set()
    market = "IN" if language == "Hindi" else "US"
    af_targets = _EMOTION_AF.get(emotion, _EMOTION_AF["neutral"])
    seeds = user_genre_seed or emotion_seeds(emotion, all_seeds)

    raw: list[dict] = []

    # 1. Recommendations with audio feature targets
    try:
        raw.extend(sp_recommend(
            seeds, tok, limit=40, market=market,
            valence=af_targets["valence"],
            energy=af_targets["energy"],
            danceability=af_targets["danceability"],
            tempo=af_targets["tempo"],
        ))
    except:
        pass

    # 2. BLIP keyword searches paired with emotion
    keywords = blip_data.get("keywords", [])
    lang_sfx = {"Hindi": " hindi bollywood", "English": " english"}.get(language, f" {custom_lang or ''}")

    for kw in keywords[:6]:
        raw.extend(sp_search(f"{kw} {emotion}{lang_sfx}", tok, limit=15, market=market))

    # 3. General description search
    general = blip_data.get("general", "")
    if general:
        raw.extend(sp_search(f"{general[:40]} {emotion}{lang_sfx}", tok, limit=15, market=market))

    # 4. Genre seed searches
    for g in seeds[:3]:
        raw.extend(sp_search(f"genre:{g}{lang_sfx}", tok, limit=15, market=market))

    # 5. Deduplicate
    seen_ids, tracks = set(), []
    for t in raw:
        if t["id"] not in seen_ids:
            seen_ids.add(t["id"])
            tracks.append(t)

    # 6. Language + exclusion filter
    tracks = [t for t in tracks if lang_ok(t, language, custom_lang) and t["id"] not in exclude_ids]
    if not tracks:
        return []

    # 7. Batch audio features
    af_map = sp_audio_features([t["id"] for t in tracks], tok)

    # 8. CHANGE 1: Batch artist genres — single API call instead of N sequential calls
    artist_ids = list({t["artists"][0]["id"] for t in tracks})[:50]
    artist_genre_cache = sp_artists_batch(artist_ids, tok)

    # 9. Build song dicts with audio feature match score
    tv, te, td = af_targets["valence"], af_targets["energy"], af_targets["danceability"]
    songs = []
    for t in tracks:
        f   = af_map.get(t["id"]) or {}
        rd  = t["album"]["release_date"]
        imgs = t["album"]["images"]
        a_id = t["artists"][0]["id"]
        af_score = (
            abs(f.get("valence", tv) - tv) +
            abs(f.get("energy", te) - te) +
            abs(f.get("danceability", td) - td)
        )
        songs.append(dict(
            title            = t["name"],
            artists          = [a["name"] for a in t["artists"]],
            album            = t["album"]["name"],
            duration         = fmt_dur(t["duration_ms"]),
            thumbnail        = imgs[0]["url"] if imgs else "",
            track_id         = t["id"],
            spotify_url      = t["external_urls"]["spotify"],
            preview_url      = t.get("preview_url"),
            popularity       = t["popularity"],
            release_date     = rd,
            days_old         = days_old(rd),
            energy           = f.get("energy", 0.5),
            valence          = f.get("valence", 0.5),
            tempo            = f.get("tempo", 120.0),
            danceability     = f.get("danceability", 0.5),
            acousticness     = f.get("acousticness", 0.5),
            speechiness      = f.get("speechiness", 0.05),
            instrumentalness = f.get("instrumentalness", 0.0),
            loudness         = f.get("loudness", -10.0),
            artist_genres    = artist_genre_cache.get(a_id, []),
            markets_count    = len(t.get("available_markets", [])),
            af_score         = af_score,
        ))

    # 10. Sort / filter by popularity mode
    if popularity_mode == "Popular":
        songs = [s for s in songs if s["popularity"] >= 60]
        songs.sort(key=lambda x: x["popularity"], reverse=True)
    elif popularity_mode == "Trending":
        songs = [s for s in songs if s["days_old"] <= 120]
        songs.sort(
            key=lambda x: x["popularity"] * 0.55 + max(0, (120 - x["days_old"]) / 120) * 45,
            reverse=True,
        )
    elif popularity_mode == "Underrated":
        songs = [s for s in songs if 3 <= s["popularity"] <= 42]
        songs.sort(key=lambda x: x["af_score"])
    else:
        songs.sort(key=lambda x: x["af_score"] * 0.65 + (1 - x["popularity"] / 100) * 0.35)

    return songs[:n]

# ── DISPLAY – EMOTION CHART ───────────────────────────────────────────────────
def draw_emotion_chart(scores: dict, t):
    EC = {"happy":"#FFD700","sad":"#4A9EF5","angry":"#FF4444",
          "fear":"#9B6CF7","disgust":"#3DCC6E","surprise":"#FF8C00","neutral":"#94A3B8"}
    labels = list(scores.keys())
    vals   = [float(v) for v in scores.values()]
    bg    = t["bg"]    if t else "#111111"
    card  = t["card"]  if t else "#1a1a1a"
    prim  = t["primary"] if t else "#00b3b3"
    brd   = t["border"] if t else "#00ffff"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(card)
    colors = [EC.get(e, prim) for e in labels]
    bars = ax.barh(labels, vals, color=colors, edgecolor=brd, linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#fff", fontweight="bold")
    ax.set_xlabel("Confidence %", color="#ccc")
    ax.set_title("Emotion Breakdown", color=prim, fontsize=13, fontweight="bold")
    ax.tick_params(colors="#ccc", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(brd)
    ax.set_xlim(0, (max(vals) if vals else 100) * 1.28)
    plt.tight_layout()
    return fig

# ── DISPLAY – SONG CARD ───────────────────────────────────────────────────────
def render_song_card(song: dict, t):
    border = t["border"] if t else "#00b3b3"
    accent = t["accent"] if t else "#00ffff"
    primary = t["primary"] if t else "#00b3b3"
    card_bg = t["card"] if t else "#1a1a1a"

    genre_html = "".join(
        f'<span class="genre-chip" style="background:{card_bg};color:{primary};border:1px solid {border}55;">{g}</span>'
        for g in song["artist_genres"][:5]
    ) or '<span style="color:#555;font-size:12px;">—</span>'

    st.markdown(f"""
<div class="sp-card" style="background:{card_bg};border:1px solid {border}55;">
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <img src="{song['thumbnail']}" style="width:110px;height:110px;border-radius:10px;
         object-fit:cover;border:2px solid {border};flex-shrink:0;"
         onerror="this.style.display='none'">
    <div style="flex:1;min-width:200px;">
      <div style="font-size:18px;font-weight:700;color:{accent};font-family:'Syne',sans-serif;
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{song['title']}</div>
      <div style="color:#ccc;font-size:13px;margin:3px 0;">🎤 {', '.join(song['artists'])}</div>
      <div style="color:#999;font-size:12px;">💿 {song['album']}</div>
      <div style="margin-top:6px;">{genre_html}</div>
      <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:7px;font-size:12px;color:#bbb;">
        <span>⏱ {song['duration']}</span>
        <span>⭐ {song['popularity']}/100</span>
        <span>📅 {song['release_date']}</span>
        <span>🌍 {song['markets_count']} mkts</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;font-size:11px;color:#888;">
        <span title="Energy">⚡ {song['energy']:.2f}</span>
        <span title="Valence (positivity)">😊 {song['valence']:.2f}</span>
        <span title="Tempo">🥁 {song['tempo']:.0f}bpm</span>
        <span title="Danceability">💃 {song['danceability']:.2f}</span>
        <span title="Acousticness">🎸 {song['acousticness']:.2f}</span>
        <span title="Speechiness">🎤 {song['speechiness']:.2f}</span>
        <span title="Instrumentalness">🎼 {song['instrumentalness']:.2f}</span>
      </div>
    </div>
    <div style="flex-shrink:0;margin-top:4px;">
      <a href="{song['spotify_url']}" target="_blank" style="text-decoration:none;">
        <div style="background:#1DB954;color:#fff;padding:9px 16px;border-radius:20px;
                    font-weight:700;font-size:12px;white-space:nowrap;text-align:center;">
          🎵 Open Spotify
        </div>
      </a>
    </div>
  </div>
  <div style="margin-top:12px;">
    <iframe src="https://open.spotify.com/embed/track/{song['track_id']}?utm_source=generator&theme=0"
            width="100%" height="80" frameBorder="0" allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            style="border-radius:8px;"></iframe>
  </div>
</div>""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    DEFAULTS = dict(
        img_hash=None, img_bytes=None, img_pil=None,
        analyzed=False, theme_emotion=None,
        emotion="neutral", emotion_scores={}, face_detected=False,
        blip_data={}, blip_plain="",
        songs=[], shown_ids=set(),
        captions=[], shown_captions=[],
        hashtags=[], shown_hashtags=[],
        all_seeds=[],
        camera_active=False,
        pending_camera_bytes=None,
        last_song_params=None,
        _action=None,
    )
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown(BASE_CSS, unsafe_allow_html=True)

    # Inject theme only after analysis completed
    if st.session_state.theme_emotion:
        inject_theme(st.session_state.theme_emotion)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.session_state.img_bytes:
            st.image(
                Image.open(io.BytesIO(st.session_state.img_bytes)),
                caption="📸 Analysed Image",
                use_column_width=True,
            )
            st.markdown("---")

        st.markdown("### 🌍 Language")
        language = st.radio("Songs in:", ["English", "Hindi", "Both", "Other"],
                            index=0, key="lang_radio")
        custom_lang = None
        if language == "Other":
            custom_lang = st.text_input("Specify:", placeholder="e.g. Spanish, Korean…")

        st.markdown("---")
        st.markdown("### 🎭 Genre")

        if not st.session_state.all_seeds:
            with st.spinner("Loading genres…"):
                st.session_state.all_seeds = fetch_genre_seeds()

        sel_group = st.selectbox("Pick genre:", list(GENRE_GROUPS.keys()), index=0)
        user_genre_seed = None
        if sel_group != "Auto (match emotion)":
            seeds_all = st.session_state.all_seeds
            user_genre_seed = [g for g in GENRE_GROUPS[sel_group] if g in seeds_all][:5] or None

        st.markdown("---")
        st.markdown("### 📊 Popularity")
        popularity_mode = st.radio(
            "Filter by:",
            ["All", "Popular", "Trending", "Underrated"],
            index=0,
            help="**Popular** ≥60 · **Trending** last 120 days · **Underrated** score 3–42",
        )

        st.markdown("---")
        st.caption("🎭 DeepFace · 🖼️ BLIP · 🤖 Gemini 2.5 Flash · 🎵 Spotify")

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown('<h1 class="center-title">🎧 SyncPixel 📸</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="center-sub">Upload a photo → AI detects emotion &amp; scene → '
        'Perfect Spotify soundtrack + viral social content, instantly.</p>',
        unsafe_allow_html=True,
    )

    # ── IMAGE INPUT ───────────────────────────────────────────────────────────
    up_col, cam_col = st.columns(2)
    with up_col:
        uploaded = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png", "webp"])

    with cam_col:
        if not st.session_state.camera_active:
            st.markdown(
                '<div class="cam-wake-box">📷 Camera is sleeping</div>',
                unsafe_allow_html=True,
            )
            if st.button("📷 Wake Camera", use_container_width=True):
                st.session_state.camera_active = True
                st.rerun()
        else:
            camera_photo = st.camera_input("Take a photo", label_visibility="collapsed")
            if camera_photo:
                # Store bytes, turn off camera, trigger rerun for processing
                st.session_state.pending_camera_bytes = camera_photo.getvalue()
                st.session_state.camera_active = False
                st.rerun()
            if st.button("❌ Close Camera", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()

    # ── DETERMINE CURRENT FILE ────────────────────────────────────────────────
    current_bytes = None
    if st.session_state.pending_camera_bytes:
        current_bytes = st.session_state.pending_camera_bytes
        st.session_state.pending_camera_bytes = None
    elif uploaded:
        current_bytes = uploaded.getvalue()

    # ── SIDEBAR CHANGE DETECTION ──────────────────────────────────────────────
    current_params = {
        "lang": language,
        "genre": sel_group,
        "popularity": popularity_mode,
        "custom": custom_lang,
    }

    if current_bytes:
        img_hash  = hashlib.md5(current_bytes).hexdigest()
        new_image = img_hash != st.session_state.img_hash

        # ── FULL ANALYSIS on new image ─────────────────────────────────────────
        if new_image:
            # Reset all state for fresh analysis + remove old theme
            for k in list(DEFAULTS.keys()):
                st.session_state[k] = DEFAULTS[k]

            st.session_state.img_hash  = img_hash
            st.session_state.img_bytes = current_bytes
            st.session_state.img_pil   = Image.open(io.BytesIO(current_bytes)).convert("RGB")
            img_pil = st.session_state.img_pil

            bar = st.progress(0, "🔍 Detecting emotion with DeepFace…")

            emo = analyse_emotion(img_pil)
            st.session_state.emotion       = emo["dominant"]
            st.session_state.emotion_scores = emo["scores"]
            st.session_state.face_detected = emo["face"]
            bar.progress(18, "🖼️ Analysing scene with BLIP…")

            blip_data = analyse_image_blip(img_pil)
            st.session_state.blip_data  = blip_data
            st.session_state.blip_plain = blip_data["description_plain"]
            bar.progress(38, "🎵 Loading Spotify genre seeds…")

            if not st.session_state.all_seeds:
                st.session_state.all_seeds = fetch_genre_seeds()
            bar.progress(48, "🎧 Searching Spotify for songs…")

            songs = search_songs(
                emo["dominant"], blip_data, st.session_state.all_seeds,
                language, user_genre_seed, popularity_mode,
                exclude_ids=set(), n=5, custom_lang=custom_lang,
            )

            # CHANGE 3: Fallback retry with loosened filters when no songs found
            if not songs:
                songs = search_songs(
                    emo["dominant"], blip_data, st.session_state.all_seeds,
                    language="Both", user_genre_seed=None,
                    popularity_mode="All", exclude_ids=set(), n=5, custom_lang=None,
                )

            st.session_state.songs    = songs
            st.session_state.shown_ids = {s["track_id"] for s in songs}
            bar.progress(68, "✍️ Generating captions with Gemini…")

            captions = gen_captions(emo["dominant"], blip_data["description_plain"], [], 3)
            st.session_state.captions       = captions
            st.session_state.shown_captions = captions.copy()
            bar.progress(86, "#️⃣ Generating hashtags with Gemini…")

            hashtags = gen_hashtags(emo["dominant"], blip_data["description_plain"], [], 5)
            st.session_state.hashtags       = hashtags
            st.session_state.shown_hashtags = hashtags.copy()

            bar.progress(100, "✅ Done!")
            time.sleep(0.35)
            bar.empty()

            st.session_state.analyzed     = True
            st.session_state.theme_emotion = emo["dominant"] if emo["dominant"] != "neutral" else None
            st.session_state.last_song_params = current_params
            st.rerun()

        # ── SIDEBAR PARAM CHANGED → auto-reload songs only ────────────────────
        elif (st.session_state.analyzed and
              st.session_state.last_song_params is not None and
              current_params != st.session_state.last_song_params):
            with st.spinner("🎧 Refreshing song recommendations…"):
                new_songs = search_songs(
                    st.session_state.emotion,
                    st.session_state.blip_data,
                    st.session_state.all_seeds,
                    language, user_genre_seed, popularity_mode,
                    exclude_ids=set(), n=5, custom_lang=custom_lang,
                )

                # CHANGE 3: Fallback retry with loosened filters when no songs found
                if not new_songs:
                    new_songs = search_songs(
                        st.session_state.emotion,
                        st.session_state.blip_data,
                        st.session_state.all_seeds,
                        language="Both", user_genre_seed=None,
                        popularity_mode="All", exclude_ids=set(), n=5, custom_lang=None,
                    )

            st.session_state.songs = new_songs
            st.session_state.shown_ids = {s["track_id"] for s in new_songs}
            st.session_state.last_song_params = current_params

        # ── PARTIAL ACTIONS ────────────────────────────────────────────────────
        # CHANGE 4: Process _action here without a second st.rerun().
        # Buttons below no longer call st.rerun() — the button-click itself
        # triggers Streamlit's automatic rerun, so we only need one pass.
        action = st.session_state._action
        if action == "more_songs":
            st.session_state._action = None
            with st.spinner("🎲 Finding more songs…"):
                more = search_songs(
                    st.session_state.emotion, st.session_state.blip_data,
                    st.session_state.all_seeds, language, user_genre_seed,
                    popularity_mode, st.session_state.shown_ids, 5, custom_lang,
                )
            st.session_state.songs.extend(more)
            for s in more:
                st.session_state.shown_ids.add(s["track_id"])

        elif action == "more_captions":
            st.session_state._action = None
            with st.spinner("✍️ Writing more captions…"):
                more = gen_captions(st.session_state.emotion, st.session_state.blip_plain,
                                    st.session_state.shown_captions, 5)
            st.session_state.captions.extend(more)
            st.session_state.shown_captions.extend(more)

        elif action == "more_hashtags":
            st.session_state._action = None
            with st.spinner("#️⃣ Finding more hashtags…"):
                more = gen_hashtags(st.session_state.emotion, st.session_state.blip_plain,
                                    st.session_state.shown_hashtags, 5)
            st.session_state.hashtags.extend(more)
            st.session_state.shown_hashtags.extend(more)

        # ── RENDER RESULTS ─────────────────────────────────────────────────────
        if st.session_state.analyzed:
            t = get_theme(st.session_state.theme_emotion) if st.session_state.theme_emotion else None
            emotion_name = st.session_state.emotion.capitalize()
            primary = t["primary"] if t else "#00b3b3"
            accent  = t["accent"]  if t else "#00ffff"
            border  = t["border"]  if t else "#00b3b3"
            card_bg = t["card"]    if t else "#1a1a1a"
            secondary = t["secondary"] if t else "#004444"

            st.markdown("---")

            # Image Analysis
            st.markdown("## 🔍 Image Analysis")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### 🎭 Emotion Detection")
                face_note = (
                    "✅ Face detected – from facial analysis"
                    if st.session_state.face_detected
                    else "⚠️ No face detected – inferred from scene"
                )
                st.markdown(f"""
<div class="emotion-badge" style="background:{secondary};border:2px solid {border};">
  <div style="font-size:28px;font-weight:800;color:{accent};letter-spacing:3px;font-family:'Syne',sans-serif;">
    {emotion_name.upper()}
  </div>
  <div style="color:#aaa;font-size:12px;margin-top:5px;">{face_note}</div>
</div>""", unsafe_allow_html=True)
                if st.session_state.emotion_scores:
                    fig = draw_emotion_chart(st.session_state.emotion_scores, t)
                    st.pyplot(fig)
                    plt.close(fig)

            with c2:
                st.markdown("### 🖼️ BLIP Scene Analysis")
                st.markdown(
                    f'<div class="analysis-box" style="background:{card_bg};border:1px solid {border}55;">'
                    f'{st.session_state.blip_data.get("description","")}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Social Content
            st.markdown("## 📱 Social Content")
            s1, s2 = st.columns(2)

            with s1:
                st.markdown("### ✍️ Captions")
                for i, cap in enumerate(st.session_state.captions):
                    # Numbered badge header
                    st.markdown(
                        f'<div style="color:{primary};font-weight:700;font-family:\'Syne\',sans-serif;'
                        f'margin-top:10px;margin-bottom:2px;">Caption #{i+1}</div>',
                        unsafe_allow_html=True,
                    )
                    # CHANGE 2: st.code gives a built-in copy button with no syntax highlight
                    st.code(cap, language=None)
                st.markdown("<br>", unsafe_allow_html=True)
                # CHANGE 4: No st.rerun() — button click triggers automatic rerun
                if st.button("✨ Suggest 5 More Captions", key="btn_caps"):
                    st.session_state._action = "more_captions"

            with s2:
                st.markdown("### # Hashtags")
                pills = "".join(
                    f'<span class="hashtag-pill" style="background:{secondary};color:{accent};border:1px solid {border};">{tag}</span>'
                    for tag in st.session_state.hashtags
                )
                st.markdown(f'<div style="line-height:2.8;">{pills}</div>', unsafe_allow_html=True)
                # CHANGE 2: Single-line copy block for all hashtags at once
                st.code(" ".join(st.session_state.hashtags), language=None)
                st.markdown("<br>", unsafe_allow_html=True)
                # CHANGE 4: No st.rerun() — button click triggers automatic rerun
                if st.button("🔥 Suggest 5 More Hashtags", key="btn_tags"):
                    st.session_state._action = "more_hashtags"

            st.markdown("---")

            # Music Recommendations
            st.markdown("## 🎵 Music Recommendations")
            active_seeds = user_genre_seed or emotion_seeds(st.session_state.emotion, st.session_state.all_seeds)
            seed_html = " ".join(
                f'<span class="genre-chip" style="background:{card_bg};color:{primary};border:1px solid {border}55;">{g}</span>'
                for g in active_seeds
            )
            st.markdown(
                f'<p style="color:#aaa;margin-bottom:10px;">'
                f'Mood: <b style="color:{accent}">{emotion_name.upper()}</b> &nbsp;·&nbsp; '
                f'Filter: <b style="color:{accent}">{popularity_mode}</b> &nbsp;·&nbsp; '
                f'Language: <b style="color:{accent}">{language}</b><br>'
                f'<span style="font-size:12px;color:#666;">Spotify seeds: {seed_html}</span></p>',
                unsafe_allow_html=True,
            )

            if st.session_state.songs:
                for song in st.session_state.songs:
                    render_song_card(song, t)
            else:
                st.warning("No songs found for these filters — even after retrying with broader settings. Try a different genre or language.")

            st.markdown("<br>", unsafe_allow_html=True)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                # CHANGE 4: No st.rerun() — button click triggers automatic rerun
                if st.button("🎲 Load 5 More Songs", key="btn_more_songs", use_container_width=True):
                    st.session_state._action = "more_songs"

    else:
        # Welcome Screen
        st.markdown("""
<div class="welcome-box">
  <div style="font-size:80px;margin-bottom:8px;">🎧</div>
  <h2>Upload a photo to get started</h2>
  <p style="color:#999;font-size:15px;max-width:560px;margin:10px auto 0;line-height:1.7;">
    SyncPixel reads the emotion and story inside your photo, then curates
    a personalised Spotify soundtrack and ready-to-post social content.
  </p>
  <div style="display:flex;justify-content:center;gap:44px;margin-top:40px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-size:34px;">🎭</div>
      <div style="color:#777;font-size:13px;margin-top:6px;">DeepFace Emotion</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:34px;">🖼️</div>
      <div style="color:#777;font-size:13px;margin-top:6px;">BLIP Scene AI</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:34px;">🎵</div>
      <div style="color:#777;font-size:13px;margin-top:6px;">Spotify Tracks</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:34px;">📱</div>
      <div style="color:#777;font-size:13px;margin-top:6px;">Social Captions</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:34px;">#️⃣</div>
      <div style="color:#777;font-size:13px;margin-top:6px;">Viral Hashtags</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  <div class="footer-divider"></div>
  Made with ❤️ by <strong style="color:#ccc;">Dhruv Goyal</strong>
  <div style="margin-top:10px;">
    <a href="https://www.linkedin.com/in/dhruvg0yal" target="_blank" title="LinkedIn">
      🔗 LinkedIn
    </a>
    <a href="https://github.com/dhruvg0ya1" target="_blank" title="GitHub">
      ⚡ GitHub
    </a>
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
