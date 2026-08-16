"""Spotify Web API client: auth, seeds, recommendations, audio features."""

import streamlit as st
import requests
import time


SPOTIFY_CLIENT_ID    = st.secrets["SPOTIFY_CLIENT_ID"]

SPOTIFY_CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]


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
