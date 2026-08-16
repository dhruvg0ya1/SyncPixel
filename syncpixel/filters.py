"""Track filtering helpers: language detection, duration and recency."""

import re
from datetime import datetime


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

def fmt_dur(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"

def days_old(rd: str) -> int:
    try:
        return (datetime.now() - datetime.strptime(rd[:10], "%Y-%m-%d")).days
    except:
        return 9999
