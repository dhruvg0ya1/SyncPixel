from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import requests
import re
import time
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import json
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="SyncPixel",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for theming
st.markdown("""
<style>
    .stApp {
        background-color: #001919;
        color: #ffffff;
    }
    
    .stButton > button {
        background-color: #008c8c;
        color: #ffffff;
        border: 1px solid #00ffff;
    }
    
    .stButton > button:hover {
        background-color: #00ffff;
        color: #001919;
    }
    
    .stMetric {
        background-color: #008c8c;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #00ffff;
    }
    
    .caption-text {
        font-size: 18px;
        font-weight: bold;
        color: #00ffff;
    }
    
    .song-card {
        background-color: #008c8c;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #00ffff;
    }
    
    .emotion-bar {
        background-color: #008c8c;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Spotify API credentials
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Load image captioning model
@st.cache_resource
def load_image_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_image_captioning_model()

# Spotify API functions
@st.cache_data(ttl=3600)  # Cache token for 1 hour
def get_spotify_token():
    """Get Spotify access token using client credentials flow"""
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
    }
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        token_info = response.json()
        return token_info['access_token']
    else:
        st.error(f"Failed to get Spotify token: {response.status_code}")
        return None

def search_spotify_tracks(query, access_token, limit=20, market="US"):
    """Search for tracks on Spotify"""
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": query,
        "type": "track",
        "limit": limit,
        "market": market
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to search Spotify: {response.status_code}")
        return None

def get_track_audio_features(track_ids, access_token):
    """Get audio features for tracks (tempo, energy, valence, etc.)"""
    url = "https://api.spotify.com/v1/audio-features"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "ids": ",".join(track_ids)
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_artist_info(artist_id, access_token):
    """Get artist information"""
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Helper functions
def get_image_caption(image):
    """Generate a caption for the image using BLIP model"""
    inputs = processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_length=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def analyze_emotion_from_face(image):
    """Analyze facial expressions and emotions using DeepFace"""
    try:
        result = DeepFace.analyze(
            img_path=np.array(image), 
            actions=['emotion', 'gender'],
            enforce_detection=False
        )
        
        if isinstance(result, list):
            result = result[0]
            
        dominant_emotion = result['dominant_emotion']
        emotions = result['emotion']
        gender = result['gender']
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotions': emotions,
            'gender': gender,
            'face_detected': True
        }
    except Exception as e:
        st.warning(f"No face detected or error in face analysis: {e}")
        return {
            'dominant_emotion': 'neutral',
            'emotions': {'neutral': 100},
            'gender': 'Unknown',
            'face_detected': False
        }

def determine_emotion_from_image(caption, face_analysis):
    """Determine emotion from image caption when no face is detected"""
    emotion_keywords = {
        'happy': ['happy', 'smile', 'laugh', 'joy', 'fun', 'bright', 'sunny', 'celebration', 'party'],
        'sad': ['sad', 'gloomy', 'rain', 'tear', 'dark', 'alone', 'lonely', 'night', 'depressed', 'worried', 'stressed'],
        'angry': ['angry', 'storm', 'fire', 'intense', 'red', 'dark', 'frustration'],
        'fear': ['scary', 'dark', 'night', 'shadow', 'fog', 'mist', 'afraid', 'anxious'],
        'surprise': ['surprise', 'unusual', 'unique', 'colorful', 'bright'],
        'neutral': ['calm', 'serene', 'peaceful', 'quiet', 'still', 'natural', 'landscape']
    }
    
    # If face was detected, use that emotion
    if face_analysis.get('face_detected', False):
        return face_analysis['dominant_emotion']
        
    # Otherwise analyze caption
    caption = caption.lower()
    emotion_scores = {emotion: 0 for emotion in emotion_keywords}
    
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in caption:
                emotion_scores[emotion] += 1
    
    # Get emotion with highest score
    if max(emotion_scores.values()) > 0:
        return max(emotion_scores, key=emotion_scores.get)
    else:
        return "neutral"

def detect_objects_from_caption(caption):
    """Extract key objects and themes from image caption"""
    words = caption.lower().split()
    stop_words = ["a", "the", "is", "in", "on", "with", "and", "of", "to", "at"]
    important_words = [word for word in words if word not in stop_words]
    
    return important_words

def generate_search_queries(analysis_results):
    """Generate search queries based on emotion or objects detected"""
    caption = analysis_results.get('caption', '')
    face_analysis = analysis_results.get('face_analysis', {})
    
    # Get emotion either from face or from image content
    emotion = determine_emotion_from_image(caption, face_analysis)
    
    # Map emotions to music emotions and audio features
    emotion_to_music = {
        'happy': {
            'genres': ['pop', 'dance', 'funk', 'disco'],
            'keywords': ['upbeat', 'happy', 'cheerful', 'joyful', 'energetic', 'celebration'],
            'valence': 'high',  # Musical positivity
            'energy': 'high'
        },
        'sad': {
            'genres': ['indie', 'alternative', 'folk', 'acoustic'],
            'keywords': ['melancholic', 'sad', 'emotional', 'heartbreak', 'soulful', 'blues'],
            'valence': 'low',
            'energy': 'low'
        },
        'angry': {
            'genres': ['rock', 'metal', 'punk', 'hardcore'],
            'keywords': ['intense', 'angry', 'powerful', 'energetic', 'rage', 'aggressive'],
            'valence': 'low',
            'energy': 'high'
        },
        'disgust': {
            'genres': ['alternative', 'grunge', 'industrial'],
            'keywords': ['dark', 'intense', 'rebellious', 'alternative', 'raw'],
            'valence': 'low',
            'energy': 'medium'
        },
        'fear': {
            'genres': ['ambient', 'electronic', 'darkwave'],
            'keywords': ['atmospheric', 'suspenseful', 'dramatic', 'haunting', 'mysterious'],
            'valence': 'low',
            'energy': 'low'
        },
        'surprise': {
            'genres': ['electronic', 'experimental', 'pop'],
            'keywords': ['exciting', 'surprising', 'dynamic', 'uplifting', 'unexpected'],
            'valence': 'high',
            'energy': 'high'
        },
        'neutral': {
            'genres': ['indie', 'chill', 'lo-fi', 'ambient'],
            'keywords': ['relaxing', 'ambient', 'chill', 'smooth', 'calm'],
            'valence': 'medium',
            'energy': 'medium'
        }
    }
    
    queries = []
    music_info = emotion_to_music.get(emotion, emotion_to_music['neutral'])
    
    # If face detected, focus on emotions
    if face_analysis.get('face_detected', True):
        # Generate genre-based queries
        for genre in music_info['genres'][:2]:
            queries.append(f"genre:{genre}")
        
        # Generate keyword-based queries
        for keyword in music_info['keywords'][:3]:
            queries.append(keyword)
            
        # Add emotion-specific query
        queries.append(f"{emotion} emotion")
    
    # If no face detected, focus on objects and context
    else:
        objects = detect_objects_from_caption(caption)
        
        if objects:
            # Use the most important objects for queries
            main_objects = objects[:2]
            for obj in main_objects:
                queries.append(obj)
            
            # Combine with emotion
            queries.append(f"{emotion} {' '.join(main_objects[:2])}")
        else:
            # Fallback to emotion-based
            for keyword in music_info['keywords'][:3]:
                queries.append(keyword)
    
    # Ensure we have enough unique queries
    queries = list(set(queries))[:5]
    
    return queries, emotion, music_info

def format_duration(duration_ms):
    """Convert milliseconds to MM:SS format"""
    seconds = duration_ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

def is_english_song(track_name, artists, album_name):
    """Check if a song is likely English based on various factors"""
    # Common English words and patterns
    english_patterns = [
        r'[a-zA-Z]',  # Contains English letters
        r'\b(the|and|of|in|to|for|with|on|at|by|from|up|about|into|over|after)\b',  # Common English words
        r'\b(love|heart|time|life|world|night|day|man|woman|girl|boy|good|bad|never|always)\b'  # Common English song words
    ]
    
    text_to_check = f"{track_name} {' '.join(artists)} {album_name}".lower()
    
    # Check for English patterns
    english_score = 0
    for pattern in english_patterns:
        if re.search(pattern, text_to_check):
            english_score += 1
    
    # Check for non-English scripts (basic check)
    non_english_patterns = [
        r'[\u0900-\u097F]',  # Devanagari (Hindi)
        r'[\u0980-\u09FF]',  # Bengali
        r'[\u0A00-\u0A7F]',  # Gurmukhi
        r'[\u0A80-\u0AFF]',  # Gujarati
        r'[\u0B00-\u0B7F]',  # Oriya
        r'[\u0B80-\u0BFF]',  # Tamil
        r'[\u0C00-\u0C7F]',  # Telugu
        r'[\u0C80-\u0CFF]',  # Kannada
        r'[\u0D00-\u0D7F]',  # Malayalam
    ]
    
    for pattern in non_english_patterns:
        if re.search(pattern, text_to_check):
            return False
    
    return english_score > 0

def is_hindi_song(track_name, artists, album_name):
    """Check if a song is likely Hindi/Bollywood"""
    text_to_check = f"{track_name} {' '.join(artists)} {album_name}".lower()
    
    # Check for Devanagari script
    if re.search(r'[\u0900-\u097F]', text_to_check):
        return True
    
    # Check for common Hindi/Bollywood keywords
    hindi_keywords = [
        'hindi', 'bollywood', 'desi', 'bhangra', 'punjabi', 'tamil', 'telugu', 'kannada', 'malayalam',
        'arijit', 'rahat', 'shreya', 'sunidhi', 'sonu', 'kumar', 'udit', 'alka', 'lata', 'kishore',
        'ar rahman', 'vishal', 'shekhar', 'shankar', 'ehsaan', 'loy', 'ilaiyaraaja', 'harris',
        'bollywood', 'playback', 'kumar sanu', 'mohammed rafi', 'mukesh'
    ]
    
    for keyword in hindi_keywords:
        if keyword in text_to_check:
            return True
    
    return False

def search_songs(queries, language_filter='English', max_results=15, user_filters=None, current_track_ids=None):
    """Search for songs using Spotify API with proper language filtering"""
    import random
    access_token = get_spotify_token()
    if not access_token:
        return []
    
    all_songs = []
    current_track_ids = current_track_ids or set()
    
    # Determine market based on language
    market = "IN" if language_filter == "Hindi" else "US"
    
    for query in queries:
        try:
            # Add language specifics to the query
            if language_filter == 'Hindi':
                search_query = f"{query} hindi bollywood"
            elif language_filter == 'English':
                search_query = f"{query} english"
            else:
                search_query = query
            
            # Search for tracks
            search_results = search_spotify_tracks(search_query, access_token, limit=30, market=market)
            
            if search_results and 'tracks' in search_results:
                tracks = search_results['tracks']['items']
                
                # Filter tracks by language
                filtered_tracks = []
                for track in tracks:
                    track_name = track['name']
                    artists = [artist['name'] for artist in track['artists']]
                    album_name = track['album']['name']
                    
                    if language_filter == 'English':
                        if is_english_song(track_name, artists, album_name) and not is_hindi_song(track_name, artists, album_name):
                            filtered_tracks.append(track)
                    elif language_filter == 'Hindi':
                        if is_hindi_song(track_name, artists, album_name):
                            filtered_tracks.append(track)
                    elif language_filter == 'Both':
                        # Only allow English or Hindi songs
                        if (is_english_song(track_name, artists, album_name) or is_hindi_song(track_name, artists, album_name)):
                            filtered_tracks.append(track)
                    else:  # Both
                        filtered_tracks.append(track)
                
                # Get track IDs for audio features
                track_ids = [track['id'] for track in filtered_tracks if track['id'] not in current_track_ids]
                
                if track_ids:
                    # Get audio features
                    audio_features = get_track_audio_features(track_ids, access_token)
                    audio_features_dict = {}
                    if audio_features and 'audio_features' in audio_features:
                        audio_features_dict = {af['id']: af for af in audio_features['audio_features'] if af}
                
                # Process each track
                for track in filtered_tracks:
                    if track['id'] in current_track_ids:
                        continue
                    
                    # Get audio features for this track
                    track_features = audio_features_dict.get(track['id'], {})
                    
                    # Extract artists
                    artists = [artist['name'] for artist in track['artists']]
                    
                    # Get album image
                    album_image = track['album']['images'][0]['url'] if track['album']['images'] else ''
                    
                    # Format duration
                    duration = format_duration(track['duration_ms'])
                    
                    # Get popularity score
                    popularity = track['popularity']
                    
                    # Calculate release recency for trending
                    release_date = track['album']['release_date']
                    days_since_release = 0
                    try:
                        if len(release_date) == 10:  # YYYY-MM-DD format
                            release_datetime = datetime.strptime(release_date, '%Y-%m-%d')
                            days_since_release = (datetime.now() - release_datetime).days
                    except:
                        days_since_release = 9999
                    
                    # Create song entry
                    song = {
                        'title': track['name'],
                        'artists': artists,
                        'album': track['album']['name'],
                        'duration': duration,
                        'duration_ms': track['duration_ms'],
                        'thumbnail': album_image,
                        'track_id': track['id'],
                        'spotify_url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url'],
                        'query': query,
                        'popularity': popularity,
                        'release_date': release_date,
                        'days_since_release': days_since_release,
                        'audio_features': track_features
                    }
                    all_songs.append(song)
                    
        except Exception as e:
            st.error(f"Error searching with query '{query}': {str(e)}")
    
    # Remove duplicates based on track_id
    unique_songs = []
    seen_ids = set()
    for song in all_songs:
        if song['track_id'] not in seen_ids:
            unique_songs.append(song)
            seen_ids.add(song['track_id'])
    
    # Apply filters properly
    filtered_songs = unique_songs
    
    if user_filters:
        # Apply genre filters first
        if 'Hip-hop' in user_filters or 'Pop' in user_filters:
            genre_filtered = []
            for song in filtered_songs:
                audio_features = song.get('audio_features', {})
                # Use audio features to identify genres
                if 'Hip-hop' in user_filters:
                    # Hip-hop typically has high energy, low acousticness
                    energy = audio_features.get('energy', 0.5)
                    acousticness = audio_features.get('acousticness', 0.5)
                    if energy > 0.6 and acousticness < 0.3:
                        genre_filtered.append(song)
                if 'Pop' in user_filters:
                    # Pop typically has high danceability, medium energy
                    danceability = audio_features.get('danceability', 0.5)
                    energy = audio_features.get('energy', 0.5)
                    if danceability > 0.6 and energy > 0.4:
                        genre_filtered.append(song)
            
            if genre_filtered:
                filtered_songs = genre_filtered
        
        # Apply popularity-based filters
        if 'Popular' in user_filters:
            # Sort by popularity descending (high popularity = popular)
            filtered_songs = sorted(filtered_songs, key=lambda x: x.get('popularity', 0), reverse=True)
            
        elif 'Emerging artists' in user_filters:
            # Sort by popularity ascending (low popularity = emerging)
            filtered_songs = sorted(filtered_songs, key=lambda x: x.get('popularity', 0))
            
        elif 'Trending' in user_filters:
            # Sort by combination of popularity and recency
            filtered_songs = sorted(filtered_songs, key=lambda x: (
                x.get('popularity', 0) * 0.7 + 
                max(0, (30 - x.get('days_since_release', 9999)) / 30) * 0.3
            ), reverse=True)
    
    # Add a small degree of randomness (shuffle top 20 results)
    top_n = min(20, len(filtered_songs))
    if top_n > 1:
        random.seed()  # Use system time
        top_songs = filtered_songs[:top_n]
        random.shuffle(top_songs)
        filtered_songs = top_songs + filtered_songs[top_n:]
    
    return filtered_songs[:max_results]

def display_songs(songs, start_idx=0, batch_size=5):
    """Display songs in batches with embedded player"""
    end_idx = min(start_idx + batch_size, len(songs))
    current_batch = songs[start_idx:end_idx]
    
    if not current_batch:
        st.write("No more songs found matching the criteria.")
        return False
    
    for i, song in enumerate(current_batch):
        with st.container():
            st.markdown('<div class="song-card">', unsafe_allow_html=True)
            cols = st.columns([1, 3, 1])
            
            with cols[0]:
                st.image(song['thumbnail'], width=150)
                
            with cols[1]:
                st.markdown(f"### {song['title']}")
                st.write(f"**Artist(s):** {', '.join(song['artists'])}")
                st.write(f"**Album:** {song['album']}")
                st.write(f"**Duration:** {song['duration']}")
                st.write(f"**Popularity:** {song['popularity']}/100")
                st.write(f"**Release Date:** {song['release_date']}")
                
                # Display audio features if available
                if song['audio_features']:
                    af = song['audio_features']
                    st.write(f"**Energy:** {af.get('energy', 'N/A'):.2f}")
                    st.write(f"**Valence (Positivity):** {af.get('valence', 'N/A'):.2f}")
                    st.write(f"**Tempo:** {af.get('tempo', 'N/A'):.0f} BPM")
            
            with cols[2]:
                # Open in Spotify button
                st.markdown(f"""
                    <a href="{song['spotify_url']}" target="_blank">
                        <button style="background-color: #1DB954; color: white; padding: 10px 20px; 
                                       border: none; border-radius: 20px; cursor: pointer; 
                                       font-weight: bold; width: 100%; margin-bottom: 10px;">
                            🎵 Open in Spotify
                        </button>
                    </a>
                """, unsafe_allow_html=True)
                
                # Preview audio - 30 second clips from Spotify
                if song['preview_url']:
                    st.write("**30-second Preview:**")
                    st.audio(song['preview_url'], format='audio/mp3')
                    
                    # Embedded Spotify player
                    st.markdown(f"""
                        <iframe src="https://open.spotify.com/embed/track/{song['track_id']}" 
                                width="100%" height="152" frameborder="0" allowtransparency="true" 
                                allow="encrypted-media"></iframe>
                    """, unsafe_allow_html=True)
                else:
                    # No preview, but still show embedded player
                    st.markdown(f"""
                        <iframe src="https://open.spotify.com/embed/track/{song['track_id']}" 
                                width="100%" height="152" frameborder="0" allowtransparency="true" 
                                allow="encrypted-media"></iframe>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            # Removed st.divider() to eliminate teal bar
    
    return True

# Main app
def main():
    st.title("🎧 SyncPixel 📸")
    st.write("Upload an image and get music recommendations based on the emotions and objects detected!")
    
    # Initialize session state
    if 'displayed_songs' not in st.session_state:
        st.session_state.displayed_songs = []
    
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = 0
    
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    
    if 'detected_emotion' not in st.session_state:
        st.session_state.detected_emotion = ""
    
    if 'music_info' not in st.session_state:
        st.session_state.music_info = {}
    
    if 'last_search_params' not in st.session_state:
        st.session_state.last_search_params = {}
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Language Preferences")
        language_option = st.radio(
            "Select language preference for songs:",
            options=["English", "Hindi", "Both"],
            index=0
        )
        
        st.header("Filters")
        filter_options = st.multiselect(
            "Select filters to customize your recommendations:",
            ["Emerging artists", "Popular", "Trending", "Hip-hop", "Pop"],
            default=[]
        )
        
        st.header("About")
        st.markdown("""
        ### SyncPixel: Music from Images
        
        SyncPixel analyzes your uploaded images to recommend music that matches the emotions and context using Spotify's vast music library.
        
        **Features:**
        - 🎭 Facial emotion analysis
        - 🧠 Image context understanding
        - 🎵 Spotify music recommendations
        - 🎼 Audio feature analysis
        - 🌍 Multi-language support
        - 🎧 30-second previews
        - ▶️ Embedded Spotify player
        
        **How it works:**
        1. Upload an image
        2. AI analyzes emotions and context
        3. Generates music queries
        4. Finds matching songs on Spotify
        5. Displays with audio features and playback
        
        **Preview Info:**
        The preview is a 30-second audio clip provided by Spotify to let you sample the song before opening it in the Spotify app.
        """)
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        with st.spinner("Analyzing image and finding music..."):
            progress_bar = st.progress(0)
            
            # Step 1: Generate caption
            progress_bar.progress(20)
            caption = get_image_caption(image)
            
            # Step 2: Analyze face for emotions
            progress_bar.progress(40)
            face_analysis = analyze_emotion_from_face(image)
            
            # Combine analysis results
            analysis_results = {
                'caption': caption,
                'face_analysis': face_analysis
            }
            
            # Step 3: Generate search queries
            progress_bar.progress(60)
            queries, detected_emotion, music_info = generate_search_queries(analysis_results)
            
            # Store in session state
            st.session_state.queries = queries
            st.session_state.detected_emotion = detected_emotion
            st.session_state.music_info = music_info
            
            # Step 4: Search for songs
            progress_bar.progress(80)
            initial_songs = search_songs(queries, language_option, max_results=10, user_filters=filter_options)
            
            # Store songs in session state
            st.session_state.displayed_songs = initial_songs
            st.session_state.current_batch = 0
            st.session_state.last_search_params = {
                'language': language_option,
                'filters': filter_options.copy()
            }
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
        
        # Display analysis results
        with col2:
            st.subheader("Image Analysis")
            
            # Display caption with larger font
            st.markdown(f'<div class="caption-text">Caption: {caption}</div>', unsafe_allow_html=True)
            
            # Display detected emotion
            st.metric("Dominant Emotion", detected_emotion.capitalize())
            
            # Emotion analysis
            st.write("**Emotion Analysis:**")
            if face_analysis['face_detected']:
                emotion_data = face_analysis['emotions']
                
                # Create emotion bar chart with app colors
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#001919')
                ax.set_facecolor('#001919')
                
                emotions = list(emotion_data.keys())
                values = list(emotion_data.values())
                
                # Use app theme colors
                colors = {
                    'happy': '#00ffff',
                    'sad': '#008c8c',
                    'angry': '#ff6b6b',
                    'fear': '#9b59b6',
                    'disgust': '#2ecc71',
                    'surprise': '#f39c12',
                    'neutral': '#95a5a6'
                }
                
                bar_colors = [colors.get(emotion, '#008c8c') for emotion in emotions]
                bars = ax.barh(emotions, values, color=bar_colors)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1f}%', va='center', fontsize=10, color='#ffffff')
                
                ax.set_xlabel('Probability %', color='#ffffff')
                ax.set_title('Detected Emotions', color='#ffffff')
                ax.tick_params(colors='#ffffff')
                ax.spines['bottom'].set_color('#ffffff')
                ax.spines['top'].set_color('#ffffff')
                ax.spines['right'].set_color('#ffffff')
                ax.spines['left'].set_color('#ffffff')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.info("No faces detected. Music recommendations based on image context.")
                objects = detect_objects_from_caption(caption)
                st.write(f"**Key elements:** {', '.join(objects[:5])}")
        
        # Display music recommendations
        st.subheader("🎵 Recommended Songs from Spotify")
        
        if len(st.session_state.displayed_songs) == 0:
            st.warning("No songs found. Try different language settings or upload another image.")
        else:
            # Display songs
            display_songs(st.session_state.displayed_songs, st.session_state.current_batch, 5)
            
            # Load new button - replaces previous 5 songs with 5 new ones
            if len(st.session_state.displayed_songs) > 5:
                if st.button("🔄 Load 5 New Songs", key=f"load_new_{st.session_state.current_batch}"):
                    # Get IDs of songs already displayed
                    current_track_ids = set()
                    # Use stored search parameters
                    search_params = st.session_state.last_search_params
                    # Fetch new songs (excluding currently displayed ones)
                    more_songs = search_songs(
                        st.session_state.queries, 
                        search_params.get('language', language_option), 
                        max_results=5,
                        user_filters=search_params.get('filters', filter_options),
                        current_track_ids=current_track_ids
                    )
                    # Replace displayed songs with new batch
                    st.session_state.displayed_songs = more_songs
                    st.session_state.current_batch = 0
                    st.rerun()

if __name__ == "__main__":
    main()