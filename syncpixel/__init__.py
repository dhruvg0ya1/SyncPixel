"""SyncPixel: emotion-aware, cross-modal music recommendation."""

from .theme import get_theme, inject_theme
from .vision import blip_caption, blip_vqa, analyse_image_blip, analyse_emotion
from .llm import gemini_text, gen_captions, gen_hashtags
from .spotify import spotify_token, fetch_genre_seeds, sp_search, sp_audio_features
from .filters import lang_ok, fmt_dur, days_old
from .recommend import emotion_seeds, search_songs
from .ui import draw_emotion_chart, render_song_card
