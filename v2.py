import os
from dotenv import load_dotenv
load_dotenv()
# ═══════════════════════════════════════════════════════════════════════════════
#  SyncPixel – Music from Images
#  Stack : Streamlit · DeepFace · Google Gemini 2.5 Flash · Spotify Web API
# ═══════════════════════════════════════════════════════════════════════════════

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

# ──────────────────────────────────────────────────────────────────────────────
#  CREDENTIALS
# ──────────────────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SyncPixel",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  EMOTION THEMES
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "happy":    dict(bg="#100e00", primary="#FFD700", secondary="#b38f00",
                     card="#1e1a00", border="#FFD700", accent="#FFE97A"),
    "sad":      dict(bg="#00080f", primary="#3B82F6", secondary="#1E3A8A",
                     card="#00122a", border="#3B82F6", accent="#93C5FD"),
    "angry":    dict(bg="#110000", primary="#EF4444", secondary="#7f1d1d",
                     card="#220000", border="#EF4444", accent="#FCA5A5"),
    "fear":     dict(bg="#080014", primary="#8B5CF6", secondary="#4C1D95",
                     card="#13002e", border="#8B5CF6", accent="#C4B5FD"),
    "disgust":  dict(bg="#001200", primary="#22C55E", secondary="#14532d",
                     card="#002200", border="#22C55E", accent="#86EFAC"),
    "surprise": dict(bg="#0f0800", primary="#F97316", secondary="#9a3412",
                     card="#1f1000", border="#F97316", accent="#FED7AA"),
    "neutral":  dict(bg="#001919", primary="#00b3b3", secondary="#005555",
                     card="#002828", border="#00ffff", accent="#00ffff"),
}

def theme(emotion: str) -> dict:
    return THEMES.get(emotion.lower(), THEMES["neutral"])

def inject_css(emotion: str):
    t = theme(emotion)
    st.markdown(f"""
<style>
.stApp {{background-color:{t['bg']};color:#f0f0f0;}}
section[data-testid="stSidebar"] {{background-color:{t['card']};}}
h1,h2,h3,h4 {{color:{t['primary']};}}
.stButton>button {{
  background-color:{t['secondary']};color:#fff;
  border:1px solid {t['border']};border-radius:8px;
  font-weight:600;transition:all .2s;
}}
.stButton>button:hover {{background-color:{t['primary']};color:#000;border-color:{t['primary']};}}
.stProgress>div>div {{background-color:{t['primary']};}}
.stRadio>div, .stSelectbox>div, label {{color:#e0e0e0 !important;}}
.sp-card {{
  background:{t['card']};border:1px solid {t['border']};
  border-radius:14px;padding:18px;margin:10px 0;
}}
.caption-box {{
  background:{t['card']};border-left:4px solid {t['primary']};
  border-radius:8px;padding:12px 16px;margin:8px 0;
  color:#f0f0f0;font-size:15px;
}}
.hashtag-pill {{
  display:inline-block;background:{t['secondary']};color:{t['accent']};
  border:1px solid {t['border']};border-radius:20px;
  padding:5px 14px;margin:4px;font-size:14px;font-weight:600;letter-spacing:.5px;
}}
.analysis-box {{
  background:{t['card']};border:1px solid {t['border']};border-radius:12px;
  padding:16px;max-height:460px;overflow-y:auto;
  color:#ddd;font-size:14px;line-height:1.75;white-space:pre-wrap;
}}
.emotion-badge {{
  background:{t['secondary']};border:2px solid {t['border']};border-radius:12px;
  padding:14px;text-align:center;margin-bottom:12px;
}}
.welcome-box {{text-align:center;padding:60px 20px;}}
.feat-icon {{font-size:40px;}}
.genre-chip {{
  display:inline-block;background:{t['card']};color:{t['primary']};
  border:1px solid {t['border']};border-radius:16px;
  padding:3px 10px;margin:3px;font-size:12px;
}}
.cam-wake {{
  background:#1a1a1a;border:2px dashed #444;border-radius:12px;
  padding:40px 20px;text-align:center;cursor:pointer;color:#888;
  font-size:15px;margin-bottom:10px;
}}
    """GET /v1/recommendations/available-genre-seeds"""
    tok = spotify_token()
    if not tok:
        return []
    r = requests.get(
        "https://api.spotify.com/v1/recommendations/available-genre-seeds",
        headers=_sp_hdr(tok), timeout=10,
    )
    return r.json().get("genres", []) if r.ok else []
    
# ──────────────────────────────────────────────────────────────────────────────
#  EMOTION → GENRE MAPPING  (uses real Spotify genre seeds)
# ──────────────────────────────────────────────────────────────────────────────
# These are Spotify genre-seed keywords; we intersect them with the live list.
_EMOTION_GENRE_HINTS = {
    "happy":    ["pop","dance","funk","disco","happy","party","summer","tropical"],
    "sad":      ["indie","folk","acoustic","blues","sad","emo","singer-songwriter","piano"],
    "angry":    ["rock","metal","punk","hardcore","grunge","heavy-metal","industrial"],
    "disgust":  ["alternative","grunge","industrial","noise","post-punk"],
    "fear":     ["ambient","darkwave","gothic","horror","post-rock","atmospheric"],
    "surprise": ["electronic","edm","experimental","pop","synthpop","indie-pop"],
    "neutral":  ["chill","lo-fi","indie","study","sleep","rainy-day","soft-rock"],
}

# Audio-feature targets per emotion (used in /recommendations)
_EMOTION_AF = {
    "happy":    dict(valence=0.82, energy=0.80, danceability=0.75),
    "sad":      dict(valence=0.18, energy=0.28, danceability=0.35),
    "angry":    dict(valence=0.18, energy=0.92, danceability=0.50),
    "disgust":  dict(valence=0.22, energy=0.65, danceability=0.45),
    "fear":     dict(valence=0.18, energy=0.38, danceability=0.30),
    "surprise": dict(valence=0.72, energy=0.85, danceability=0.78),
    "neutral":  dict(valence=0.50, energy=0.48, danceability=0.55),
}

def emotion_genre_seeds(emotion: str, all_seeds: list[str]) -> list[str]:
    """Return up to 5 valid Spotify genre seeds for the given emotion."""
    hints = _EMOTION_GENRE_HINTS.get(emotion, _EMOTION_GENRE_HINTS["neutral"])
    matched = [g for g in hints if g in all_seeds]
    if not matched:
        matched = [g for g in all_seeds if any(h in g for h in hints)]
    return matched[:5] if matched else all_seeds[:5]

# ──────────────────────────────────────────────────────────────────────────────
#  SPOTIFY – SEARCH  (GET /v1/search)
# ──────────────────────────────────────────────────────────────────────────────
def sp_search(query: str, tok: str, limit=25, market="US") -> dict | None:
    r = requests.get(
        "https://api.spotify.com/v1/search",
        headers=_sp_hdr(tok),
        params={"q": query, "type": "track", "limit": limit, "market": market},
        timeout=10,
    )
    return r.json() if r.ok else None

# ──────────────────────────────────────────────────────────────────────────────
#  SPOTIFY – RECOMMENDATIONS  (GET /v1/recommendations)
# ──────────────────────────────────────────────────────────────────────────────
def sp_recommendations(seed_genres: list[str], tok: str,
                        limit=30, market="US",
                        target_valence=None, target_energy=None,
                        target_danceability=None) -> dict | None:
    params: dict = {
        "seed_genres": ",".join(seed_genres[:5]),
        "limit": limit,
        "market": market,
    }
    if target_valence    is not None: params["target_valence"]     = target_valence
    if target_energy     is not None: params["target_energy"]      = target_energy
    if target_danceability is not None: params["target_danceability"] = target_danceability
    r = requests.get(
        "https://api.spotify.com/v1/recommendations",
        headers=_sp_hdr(tok), params=params, timeout=10,
    )
    return r.json() if r.ok else None

# ──────────────────────────────────────────────────────────────────────────────
#  SPOTIFY – AUDIO FEATURES  (GET /v1/audio-features)
# ──────────────────────────────────────────────────────────────────────────────
def sp_audio_features(ids: list[str], tok: str) -> dict:
    if not ids:
        return {}
    r = requests.get(
        "https://api.spotify.com/v1/audio-features",
        headers=_sp_hdr(tok),
        params={"ids": ",".join(ids[:100])},
        timeout=10,
    )
    if not r.ok:
        return {}
    return {f["id"]: f for f in (r.json().get("audio_features") or []) if f}

# ──────────────────────────────────────────────────────────────────────────────
#  SPOTIFY – ARTIST INFO  (GET /v1/artists/{id})
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def sp_artist_genres(artist_id: str, tok: str) -> list[str]:
    """Return genres list for a single artist."""
    r = requests.get(
        f"https://api.spotify.com/v1/artists/{artist_id}",
        headers=_sp_hdr(tok), timeout=8,
    )
    return r.json().get("genres", []) if r.ok else []

# ──────────────────────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION  (heuristic + Spotify 'available_markets')
# ──────────────────────────────────────────────────────────────────────────────
_HINDI_KW = {
    "hindi","bollywood","desi","bhangra","punjabi",
    "arijit","rahat fateh","shreya ghoshal","sunidhi","sonu nigam",
    "udit narayan","alka yagnik","lata mangeshkar","kishore kumar",
    "ar rahman","shankar ehsaan","ilaiyaraaja","kumar sanu","rafi",
    "playback","t-series","zee music","saregama",
}

def _track_text(t: dict) -> str:
    return (t["name"] + " " +
            " ".join(a["name"] for a in t["artists"]) + " " +
            t["album"]["name"]).lower()

def _has_devanagari(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", s))

def _is_hindi(t: dict) -> bool:
    txt = _track_text(t)
    if _has_devanagari(txt):
        return True
    return any(kw in txt for kw in _HINDI_KW)

def _is_english(t: dict) -> bool:
    txt = _track_text(t)
    # Reject if contains non-Latin scripts (CJK, Cyrillic, Devanagari …)
    if re.search(r"[\u0900-\u097F\u4E00-\u9FFF\u3040-\u30FF\u0400-\u04FF\uAC00-\uD7AF]", txt):
        return False
    return bool(re.search(r"[a-zA-Z]", txt))

def _lang_ok(t: dict, language: str, custom_lang: str | None = None) -> bool:
    if language == "English":
        return _is_english(t) and not _is_hindi(t)
    if language == "Hindi":
        return _is_hindi(t)
    if language == "Both":
        return _is_english(t) or _is_hindi(t)
    if language == "Other" and custom_lang:
        return custom_lang.lower() in _track_text(t)
    return True   # "Other" with no spec → no filter

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def fmt_dur(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"

def _days_old(release_date: str) -> int:
    try:
        return (datetime.now() - datetime.strptime(release_date[:10], "%Y-%m-%d")).days
    except Exception:
        return 9999

# ──────────────────────────────────────────────────────────────────────────────
#  CORE – EMOTION ANALYSIS  (DeepFace)
# ──────────────────────────────────────────────────────────────────────────────
def analyse_emotion(img_pil: Image.Image) -> dict:
    try:
        res = DeepFace.analyze(
            img_path=np.array(img_pil),
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        return dict(dominant=res["dominant_emotion"],
                    scores=res["emotion"], face=True)
    except Exception:
        return dict(dominant="neutral", scores={"neutral": 100.0}, face=False)

# ──────────────────────────────────────────────────────────────────────────────
#  CORE – COMPREHENSIVE IMAGE ANALYSIS  (Gemini)
# ──────────────────────────────────────────────────────────────────────────────
def analyse_image(img_pil: Image.Image, emotion: str) -> str:
    prompt = f"""You are an expert image analyst. Analyse this photo comprehensively.
Detected dominant emotion: **{emotion.upper()}**

Provide detailed insight under EACH of these headings:
1. **Background & Setting** – environment, location, time of day, weather, season
2. **Context & Story** – what is happening, occasion, event, narrative arc
3. **Objects & Elements** – every visible item, clothing, props, food, vehicles
4. **People & Expressions** – appearance, age estimate, pose, facial expression, body language
5. **Mood & Atmosphere** – lighting quality, color palette, shadows, overall emotional vibe
6. **Themes & Symbols** – recurring motifs, cultural references, deeper meaning
7. **Scene Classification** – indoor/outdoor, urban/rural/nature, formal/casual/party/sports etc.

Be thorough and specific — this drives music recommendations and social captions."""
    return gemini_call(prompt, img_pil)

# ──────────────────────────────────────────────────────────────────────────────
#  CORE – CAPTIONS  (Gemini)
# ──────────────────────────────────────────────────────────────────────────────
def gen_captions(img_pil: Image.Image, emotion: str,
                 analysis: str, exclude: list, n: int = 3) -> list[str]:
    excl_block = (
        f"\n\nIMPORTANT – DO NOT reuse or rephrase any of these:\n" +
        "\n".join(f"- {c}" for c in exclude)
    ) if exclude else ""

    prompt = f"""Write exactly {n} catchy, viral social media captions for this image.

Image emotion: {emotion}
Image context summary: {analysis[:700]}
{excl_block}

Strict rules:
• Output ONLY bullet points, one caption per line, each starting with •
• Each caption must include relevant emojis
• Mix tones: one inspirational, one witty/playful, one emotionally resonant
• Max 160 characters per caption
• No numbering, no headers, no explanations — ONLY the bullet list"""
    raw = gemini_call(prompt, img_pil)
    caps = []
    excl_set = {c.lower() for c in exclude}
    for line in raw.splitlines():
        c = line.lstrip("•-*· ").strip()
        if c and c.lower() not in excl_set and len(c) > 10:
            caps.append(c)
    return caps[:n]

# ──────────────────────────────────────────────────────────────────────────────
#  CORE – HASHTAGS  (Gemini)
# ──────────────────────────────────────────────────────────────────────────────
def gen_hashtags(img_pil: Image.Image, emotion: str,
                 analysis: str, exclude: list, n: int = 5) -> list[str]:
    excl_block = (
        f"\nNEVER output any of these (they are already used): {' '.join(exclude)}"
    ) if exclude else ""

    prompt = f"""Generate exactly {n} trending, viral hashtags for a social media post.

Post emotion: {emotion}
Scene context: {analysis[:500]}
{excl_block}

Strict output rules:
• ONE line only
• All {n} hashtags separated by spaces
• Format: #tag1 #tag2 #tag3 ...
• No explanations, no bullet points, no extra lines
• All hashtags must be new and distinct from excluded ones"""
    raw = gemini_call(prompt, img_pil)
    tags = re.findall(r"#\w+", raw)
    seen = {e.lower() for e in exclude}
    result = []
    for tag in tags:
        if tag.lower() not in seen:
            result.append(tag)
            seen.add(tag.lower())
    return result[:n]

# ──────────────────────────────────────────────────────────────────────────────
#  CORE – SONG SEARCH  (full Spotify API)
# ──────────────────────────────────────────────────────────────────────────────
def search_songs(
    emotion: str,
    image_analysis: str,
    all_genre_seeds: list[str],
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

    # Decide genre seeds: user-chosen or emotion-mapped (from live Spotify seeds)
    seeds = user_genre_seed or emotion_genre_seeds(emotion, all_genre_seeds)

    raw_tracks: list[dict] = []

    # ── 1. Recommendations endpoint ──────────────────────────────────────────
    try:
        rec = sp_recommendations(
            seeds, tok, limit=35, market=market,
            target_valence=af_targets["valence"],
            target_energy=af_targets["energy"],
            target_danceability=af_targets["danceability"],
        )
        if rec and "tracks" in rec:
            raw_tracks.extend(rec["tracks"])
    except Exception:
        pass

    # ── 2. Search endpoint – emotion + context keywords ───────────────────────
    # Ask Gemini for 3 short music-search keywords derived from the image
    kw_prompt = (
        f"Image emotion: {emotion}. "
        f"Scene: {image_analysis[:300]}. "
        "Give me exactly 3 short Spotify search phrases (2-4 words each) "
        "to find matching music. Output only the 3 phrases, one per line, no numbering."
    )
    kw_raw = gemini_call(kw_prompt)
    kw_lines = [l.strip().lstrip("-•*1234567890. ") for l in kw_raw.splitlines() if l.strip()]
    keywords = kw_lines[:3] if kw_lines else [emotion]

    lang_suffix = {"Hindi": " hindi bollywood", "English": " english",
                   "Other": f" {custom_lang or ''}"}.get(language, "")

    for kw in keywords:
        res = sp_search(kw + lang_suffix, tok, limit=20, market=market)
        if res and "tracks" in res:
            raw_tracks.extend(res["tracks"]["items"])

    # Also search by genre seeds directly
    for g in seeds[:2]:
        res = sp_search(f"genre:{g}{lang_suffix}", tok, limit=15, market=market)
        if res and "tracks" in res:
            raw_tracks.extend(res["tracks"]["items"])

    # ── 3. Deduplicate ────────────────────────────────────────────────────────
    seen: set[str] = set()
    tracks: list[dict] = []
    for t in raw_tracks:
        if t["id"] not in seen:
            seen.add(t["id"])
            tracks.append(t)

    # ── 4. Language filter ────────────────────────────────────────────────────
    tracks = [t for t in tracks if _lang_ok(t, language, custom_lang)]
    tracks = [t for t in tracks if t["id"] not in exclude_ids]

    if not tracks:
        return []

    # ── 5. Audio features (batch) ─────────────────────────────────────────────
    af_map = sp_audio_features([t["id"] for t in tracks], tok)

    # ── 6. Artist genres (sample first-artist for each track) ─────────────────
    # We batch unique artist IDs to avoid N+1 calls
    artist_ids = list({t["artists"][0]["id"] for t in tracks})[:30]
    artist_genre_cache: dict[str, list[str]] = {}
    for aid in artist_ids:
        artist_genre_cache[aid] = sp_artist_genres(aid, tok)

    # ── 7. Build song dicts ───────────────────────────────────────────────────
    songs: list[dict] = []
    for t in tracks:
        af  = af_map.get(t["id"]) or {}
        rd  = t["album"]["release_date"]
        imgs = t["album"]["images"]
        a_id = t["artists"][0]["id"]
        songs.append(dict(
            title       = t["name"],
            artists     = [a["name"] for a in t["artists"]],
            album       = t["album"]["name"],
            duration    = fmt_dur(t["duration_ms"]),
            thumbnail   = imgs[0]["url"] if imgs else "",
            track_id    = t["id"],
            spotify_url = t["external_urls"]["spotify"],
            preview_url = t.get("preview_url"),
            popularity  = t["popularity"],
            release_date= rd,
            days_old    = _days_old(rd),
            energy      = af.get("energy",       0.5),
            valence     = af.get("valence",       0.5),
            tempo       = af.get("tempo",         120.0),
            danceability= af.get("danceability",  0.5),
            loudness    = af.get("loudness",      -10.0),
            speechiness = af.get("speechiness",   0.05),
            acousticness= af.get("acousticness",  0.5),
            instrumentalness = af.get("instrumentalness", 0.0),
            artist_genres = artist_genre_cache.get(a_id, []),
            markets_count = len(t.get("available_markets", [])),
        ))

    # ── 8. Popularity / recency filter ───────────────────────────────────────
    if popularity_mode == "Popular":
        songs = [s for s in songs if s["popularity"] >= 65]
        songs.sort(key=lambda x: x["popularity"], reverse=True)

    elif popularity_mode == "Trending":
        songs = [s for s in songs if s["days_old"] <= 120]
        songs.sort(
            key=lambda x: x["popularity"] * 0.55 +
                          max(0.0, (120 - x["days_old"]) / 120.0) * 45,
            reverse=True,
        )

    elif popularity_mode == "Underrated":
        songs = [s for s in songs if 3 <= s["popularity"] <= 42]
        tv, te = af_targets["valence"], af_targets["energy"]
        songs.sort(key=lambda x: abs(x["valence"] - tv) + abs(x["energy"] - te))

    else:  # All
        random.shuffle(songs)

    return songs[:n]

# ──────────────────────────────────────────────────────────────────────────────
#  DISPLAY – EMOTION CHART
# ──────────────────────────────────────────────────────────────────────────────
def emotion_chart(scores: dict, t: dict):
    EC = {
        "happy":"#FFD700","sad":"#3B82F6","angry":"#EF4444",
        "fear":"#8B5CF6","disgust":"#22C55E","surprise":"#F97316","neutral":"#94A3B8",
    }
    labels = list(scores.keys())
    vals   = [float(v) for v in scores.values()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["card"])
    colors = [EC.get(e, t["primary"]) for e in labels]
    bars   = ax.barh(labels, vals, color=colors, edgecolor=t["border"], linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10,
                color="#ffffff", fontweight="bold")
    ax.set_xlabel("Confidence %", color="#ccc")
    ax.set_title("Emotion Confidence", color=t["primary"], fontsize=13, fontweight="bold")
    ax.tick_params(colors="#ccc", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(t["border"])
    ax.set_xlim(0, (max(vals) if vals else 100) * 1.28)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
#  DISPLAY – SONG CARD
# ──────────────────────────────────────────────────────────────────────────────
def song_card(song: dict, t: dict):
    genre_chips = "".join(
        f'<span class="genre-chip">{g}</span>'
        for g in song["artist_genres"][:5]
    ) if song["artist_genres"] else '<span style="color:#777;font-size:12px;">No genre data</span>'

    st.markdown(f"""
<div class="sp-card">
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <img src="{song['thumbnail']}"
         style="width:120px;height:120px;border-radius:12px;
                object-fit:cover;border:2px solid {t['border']};flex-shrink:0;">
    <div style="flex:1;min-width:200px;">
      <div style="font-size:19px;font-weight:700;color:{t['accent']};
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        {song['title']}
      </div>
      <div style="color:#ccc;margin:3px 0;font-size:14px;">
        🎤 {', '.join(song['artists'])}
      </div>
      <div style="color:#aaa;font-size:13px;">💿 {song['album']}</div>
      <div style="margin-top:5px;">{genre_chips}</div>
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:7px;font-size:13px;color:#bbb;">
        <span>⏱ {song['duration']}</span>
        <span>⭐ {song['popularity']}/100</span>
        <span>📅 {song['release_date']}</span>
        <span>🌍 {song['markets_count']} markets</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:4px;font-size:12px;color:#999;">
        <span>⚡ Energy {song['energy']:.2f}</span>
        <span>😊 Valence {song['valence']:.2f}</span>
        <span>🥁 {song['tempo']:.0f} BPM</span>
        <span>💃 Dance {song['danceability']:.2f}</span>
        <span>🎸 Acoustic {song['acousticness']:.2f}</span>
        <span>🎤 Speech {song['speechiness']:.2f}</span>
      </div>
    </div>
    <div style="flex-shrink:0;margin-top:4px;">
      <a href="{song['spotify_url']}" target="_blank" style="text-decoration:none;">
        <div style="background:#1DB954;color:#fff;padding:10px 18px;
                    border-radius:20px;font-weight:700;font-size:13px;
                    white-space:nowrap;text-align:center;">
          🎵 Open in Spotify
        </div>
      </a>
    </div>
  </div>
  <div style="margin-top:10px;">
    <iframe src="https://open.spotify.com/embed/track/{song['track_id']}?utm_source=generator&theme=0"
            width="100%" height="80" frameBorder="0" allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            style="border-radius:8px;">
    </iframe>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Session-state defaults ────────────────────────────────────────────────
    _DEFAULTS = dict(
        img_hash=None, img_bytes=None, img_pil=None, analyzed=False,
        emotion="neutral", emotion_scores={}, face_detected=False,
        image_analysis="",
        songs=[], shown_song_ids=set(),
        captions=[], shown_captions=[],
        hashtags=[], shown_hashtags=[],
        all_genre_seeds=[],
        camera_active=False,
        _action=None,
    )
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    inject_css(st.session_state.emotion)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.session_state.img_bytes:
            st.image(
                Image.open(io.BytesIO(st.session_state.img_bytes)),
                caption="📸 Analysed Image",
                use_column_width=True,
            )
            st.markdown("---")

        st.markdown("### 🌍 Song Language")
        language = st.radio(
            "Preferred language:",
            ["English", "Hindi", "Both", "Other"],
            index=0, key="lang_radio",
        )
        custom_lang = None
        if language == "Other":
            custom_lang = st.text_input(
                "Specify language:", placeholder="e.g. Spanish, Korean, French…"
            )

        st.markdown("---")
        st.markdown("### 🎭 Genre")

        # Fetch live seeds; show spinner only first time
        if not st.session_state.all_genre_seeds:
            with st.spinner("Loading Spotify genres…"):
                st.session_state.all_genre_seeds = fetch_genre_seeds()

        all_seeds = st.session_state.all_genre_seeds
        genre_display = ["Auto (match emotion)"] + [
            g.replace("-", " ").title() for g in all_seeds
        ]
        sel_label = st.selectbox("Spotify genre seed:", genre_display, index=0)
        user_genre_seed: list[str] | None = None
        if sel_label != "Auto (match emotion)":
            raw_g = all_seeds[genre_display.index(sel_label) - 1]
            user_genre_seed = [raw_g]

        st.markdown("---")
        st.markdown("### 📊 Song Popularity")
        popularity_mode = st.radio(
            "Filter by:",
            ["All", "Popular", "Trending", "Underrated"],
            index=0,
            help=(
                "**Popular** – chart-toppers (score ≥ 65)\n\n"
                "**Trending** – released in last 120 days\n\n"
                "**Underrated** – hidden gems (score 3–42) with best emotion-match"
            ),
        )

        st.markdown("---")
        st.markdown("### ℹ️ SyncPixel")
        st.caption("🎭 DeepFace · 🤖 Gemini 2.5 Flash · 🎵 Spotify Web API")

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.title("🎧 SyncPixel 📸")
    st.markdown(
        '<p style="color:#aaa;font-size:16px;">'
        "Upload or capture a photo → AI detects emotion &amp; context → "
        "Perfect Spotify soundtrack + social captions, instantly.</p>",
        unsafe_allow_html=True,
    )

    # ── IMAGE INPUT ───────────────────────────────────────────────────────────
    st.markdown("### 📷 Upload or Capture")
    up_col, cam_col = st.columns(2)
    with up_col:
        uploaded = st.file_uploader(
            "Drag & drop or browse", type=["jpg", "jpeg", "png", "webp"],
        )

    with cam_col:
        # Lazy camera: show "wake" button first, then reveal camera widget
        if not st.session_state.camera_active:
            st.markdown('<div class="cam-wake">📷 Click to Wake Camera</div>',
                        unsafe_allow_html=True)
            if st.button("📷 Wake Camera", use_container_width=True):
                st.session_state.camera_active = True
                st.rerun()
            camera = None
        else:
            camera = st.camera_input("Take a photo", label_visibility="collapsed")
            if st.button("❌ Close Camera", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()

    current_file = uploaded or camera

    if current_file:
        raw_bytes = current_file.getvalue()
        img_hash  = hashlib.md5(raw_bytes).hexdigest()
        new_image = img_hash != st.session_state.img_hash

        # ── FULL ANALYSIS on new image ────────────────────────────────────────
        if new_image:
            for k in ("img_hash","img_bytes","img_pil","analyzed",
                      "emotion","emotion_scores","face_detected","image_analysis",
                      "songs","shown_song_ids","captions","shown_captions",
                      "hashtags","shown_hashtags","_action"):
                st.session_state[k] = _DEFAULTS[k]

            st.session_state.img_hash  = img_hash
            st.session_state.img_bytes = raw_bytes
            st.session_state.img_pil   = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img_pil = st.session_state.img_pil

            bar = st.progress(0, "🔍 Detecting emotion (DeepFace)…")

            # 1. DeepFace emotion
            emo = analyse_emotion(img_pil)
            st.session_state.emotion       = emo["dominant"]
            st.session_state.emotion_scores= emo["scores"]
            st.session_state.face_detected = emo["face"]
            inject_css(emo["dominant"])
            bar.progress(18, "🤖 Gemini comprehensive image analysis…")

            # 2. Gemini image analysis
            analysis = analyse_image(img_pil, emo["dominant"])
            st.session_state.image_analysis = analysis
            bar.progress(40, "🎵 Fetching Spotify genre seeds…")

            # 3. Ensure genre seeds loaded
            if not st.session_state.all_genre_seeds:
                st.session_state.all_genre_seeds = fetch_genre_seeds()
            bar.progress(50, "🎧 Searching Spotify for songs…")

            # 4. Songs
            songs = search_songs(
                emo["dominant"], analysis,
                st.session_state.all_genre_seeds,
                language, user_genre_seed, popularity_mode,
                exclude_ids=set(), n=5, custom_lang=custom_lang,
            )
            st.session_state.songs          = songs
            st.session_state.shown_song_ids = {s["track_id"] for s in songs}
            bar.progress(70, "✍️ Generating captions (Gemini)…")

            # 5. Captions
            captions = gen_captions(img_pil, emo["dominant"], analysis, exclude=[], n=3)
            st.session_state.captions      = captions
            st.session_state.shown_captions= captions.copy()
            bar.progress(86, "# Generating hashtags (Gemini)…")

            # 6. Hashtags
            hashtags = gen_hashtags(img_pil, emo["dominant"], analysis, exclude=[], n=5)
            st.session_state.hashtags      = hashtags
            st.session_state.shown_hashtags= hashtags.copy()

            bar.progress(100, "✅ Done!")
            time.sleep(0.3)
            bar.empty()
            st.session_state.analyzed = True

        # ── PARTIAL ACTIONS (no re-analysis) ─────────────────────────────────
        action = st.session_state._action
        if action == "more_songs":
            st.session_state._action = None
            new_songs = search_songs(
                st.session_state.emotion,
                st.session_state.image_analysis,
                st.session_state.all_genre_seeds,
                language, user_genre_seed, popularity_mode,
                exclude_ids=st.session_state.shown_song_ids,
                n=5, custom_lang=custom_lang,
            )
            st.session_state.songs.extend(new_songs)
            for s in new_songs:
                st.session_state.shown_song_ids.add(s["track_id"])

        elif action == "more_captions":
            st.session_state._action = None
            new_caps = gen_captions(
                st.session_state.img_pil,
                st.session_state.emotion,
                st.session_state.image_analysis,
                exclude=st.session_state.shown_captions,
                n=5,
            )
            st.session_state.captions.extend(new_caps)
            st.session_state.shown_captions.extend(new_caps)

        elif action == "more_hashtags":
            st.session_state._action = None
            new_tags = gen_hashtags(
                st.session_state.img_pil,
                st.session_state.emotion,
                st.session_state.image_analysis,
                exclude=st.session_state.shown_hashtags,
                n=5,
            )
            st.session_state.hashtags.extend(new_tags)
            st.session_state.shown_hashtags.extend(new_tags)

        # ── RENDER ────────────────────────────────────────────────────────────
        if st.session_state.analyzed:
            t            = theme(st.session_state.emotion)
            emotion_name = st.session_state.emotion.capitalize()

            st.markdown("---")

            # ── ANALYSIS SECTION ─────────────────────────────────────────────
            st.markdown("## 🔍 Image Analysis")
            ac1, ac2 = st.columns([1, 1])

            with ac1:
                st.markdown("### 🎭 Emotion Detection")
                face_note = (
                    "✅ Face detected – emotion from facial analysis"
                    if st.session_state.face_detected
                    else "⚠️ No face – emotion inferred from scene context"
                )
                st.markdown(f"""
<div class="emotion-badge">
  <div style="font-size:28px;font-weight:800;color:{t['accent']};letter-spacing:2px;">
    {emotion_name.upper()}
  </div>
  <div style="color:#aaa;font-size:13px;margin-top:5px;">{face_note}</div>
</div>""", unsafe_allow_html=True)
                if st.session_state.emotion_scores:
                    fig = emotion_chart(st.session_state.emotion_scores, t)
                    st.pyplot(fig)
                    plt.close(fig)

            with ac2:
                st.markdown("### 🤖 Gemini AI Analysis")
                st.markdown(
                    f'<div class="analysis-box">{st.session_state.image_analysis}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── SOCIAL MEDIA SECTION ─────────────────────────────────────────
            st.markdown("## 📱 Social Media Content")
            sc1, sc2 = st.columns([1, 1])

            with sc1:
                st.markdown("### ✍️ Captions")
                for i, cap in enumerate(st.session_state.captions):
                    st.markdown(f"""
<div class="caption-box">
  <span style="color:{t['primary']};font-weight:700;margin-right:6px;">#{i+1}</span>{cap}
</div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✨ Suggest 5 More Captions", key="btn_more_captions"):
                    st.session_state._action = "more_captions"
                    st.rerun()

            with sc2:
                st.markdown("### # Hashtags")
                pills = "".join(
                    f'<span class="hashtag-pill">{tag}</span>'
                    for tag in st.session_state.hashtags
                )
                st.markdown(
                    f'<div style="line-height:2.6;">{pills}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔥 Suggest 5 More Hashtags", key="btn_more_hashtags"):
                    st.session_state._action = "more_hashtags"
                    st.rerun()

            st.markdown("---")

            # ── MUSIC SECTION ─────────────────────────────────────────────────
            st.markdown("## 🎵 Music Recommendations")
            active_seeds = user_genre_seed or emotion_genre_seeds(
                st.session_state.emotion, st.session_state.all_genre_seeds
            )
            seed_chips = " &nbsp; ".join(
                f'<span class="genre-chip">{g}</span>' for g in active_seeds
            )
            st.markdown(
                f'<p style="color:#aaa;margin-bottom:10px;">'
                f'Mood: <strong style="color:{t["accent"]}">{emotion_name.upper()}</strong> &nbsp;·&nbsp; '
                f'Filter: <strong style="color:{t["accent"]}">{popularity_mode}</strong> &nbsp;·&nbsp; '
                f'Language: <strong style="color:{t["accent"]}">{language}</strong><br>'
                f'<span style="font-size:12px;color:#888;">Genre seeds: {seed_chips}</span></p>',
                unsafe_allow_html=True,
            )

            if st.session_state.songs:
                for song in st.session_state.songs:
                    song_card(song, t)
            else:
                st.warning(
                    "No songs found for the current filters. "
                    "Try changing language, genre or popularity in the sidebar."
                )

            st.markdown("<br>", unsafe_allow_html=True)
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                if st.button("🎲 Load 5 More Songs", key="btn_more_songs",
                             use_container_width=True):
                    st.session_state._action = "more_songs"
                    st.rerun()

    else:
        # ── WELCOME ───────────────────────────────────────────────────────────
        st.markdown("""
<div class="welcome-box">
  <div style="font-size:90px;">🎧</div>
  <h2 style="margin-top:10px;">Upload a photo to get started</h2>
  <p style="color:#aaa;font-size:16px;max-width:600px;margin:10px auto 0;">
    SyncPixel reads the emotion and story inside your photo, then curates
    a personalised Spotify soundtrack and ready-to-post social content.
  </p>
  <div style="display:flex;justify-content:center;gap:48px;margin-top:40px;flex-wrap:wrap;">
    <div><div class="feat-icon">🎭</div>
         <div style="color:#aaa;margin-top:6px;font-size:14px;">DeepFace Emotion</div></div>
    <div><div class="feat-icon">🤖</div>
         <div style="color:#aaa;margin-top:6px;font-size:14px;">Gemini Vision AI</div></div>
    <div><div class="feat-icon">🎵</div>
         <div style="color:#aaa;margin-top:6px;font-size:14px;">Spotify Tracks</div></div>
    <div><div class="feat-icon">📱</div>
         <div style="color:#aaa;margin-top:6px;font-size:14px;">Social Captions</div></div>
    <div><div class="feat-icon">#️⃣</div>
         <div style="color:#aaa;margin-top:6px;font-size:14px;">Viral Hashtags</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()