"""Maps an emotion to genre seeds and assembles the ranked track list."""

import streamlit as st
import random

from .spotify import (spotify_token, fetch_genre_seeds, sp_recommend, sp_search,
                      sp_audio_features, sp_artists_batch)
from .filters import lang_ok, days_old, fmt_dur


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
