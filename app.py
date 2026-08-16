"""SyncPixel - emotion-aware, cross-modal music recommendation.

Thin entry point. All logic lives in the `syncpixel` package:
    theme     colour themes per emotion
    vision    BLIP captioning/VQA + DeepFace emotion classification
    llm       Gemini caption and hashtag generation
    spotify   Spotify Web API client
    filters   language, duration and recency filters
    recommend emotion -> genre seeds -> ranked tracks
    ui        emotion chart and track cards
"""
import streamlit as st

st.set_page_config(
    page_title="SyncPixel",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

from syncpixel.theme import get_theme, inject_theme
from syncpixel.vision import analyse_image_blip, analyse_emotion
from syncpixel.llm import gen_captions, gen_hashtags
from syncpixel.recommend import search_songs
from syncpixel.ui import draw_emotion_chart, render_song_card


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
