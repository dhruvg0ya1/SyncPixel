"""Streamlit rendering: emotion chart and track cards."""

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .filters import fmt_dur



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
