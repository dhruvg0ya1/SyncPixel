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
from googleapiclient.discovery import build
import base64

# Page configuration
st.set_page_config(
    page_title="SyncPixel",
    page_icon="🎵",
    layout="wide"
)

# Initialize YouTube API
# Set your YouTube Data API key here
YOUTUBE_API_KEY = "AIzaSyBLnfhIWuUv5F7c_sLgK8IgxrdInOUErLk"  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Load image captioning model
@st.cache_resource
def load_image_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_image_captioning_model()

# Function to add background image to a specific element
def add_bg_to_element(image_path, element_selector=".stApp"):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            {element_selector} {{
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        return True
    except Exception as e:
        print(f"Error loading background image: {e}")
        return False

# Helper functions
def get_image_caption(image):
    """Generate a caption for the image using BLIP model"""
    inputs = processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_length=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def analyze_mood_from_face(image):
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

def determine_mood_from_image(caption, face_analysis):
    """Determine mood from image caption when no face is detected"""
    mood_keywords = {
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
    mood_scores = {mood: 0 for mood in mood_keywords}
    
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in caption:
                mood_scores[mood] += 1
    
    # Get mood with highest score
    if max(mood_scores.values()) > 0:
        return max(mood_scores, key=mood_scores.get)
    else:
        # Default to neutral if no matches
        return "neutral"

def detect_objects_from_caption(caption):
    """Extract key objects and themes from image caption"""
    # Simple approach: extract nouns and key descriptors from caption
    words = caption.lower().split()
    # Remove common stop words
    stop_words = ["a", "the", "is", "in", "on", "with", "and", "of", "to", "at"]
    important_words = [word for word in words if word not in stop_words]
    
    return important_words

def generate_search_queries(analysis_results):
    """Generate search queries based on mood/emotion or objects detected"""
    caption = analysis_results.get('caption', '')
    face_analysis = analysis_results.get('face_analysis', {})
    
    # Get mood either from face or from image content
    mood = determine_mood_from_image(caption, face_analysis)
    
    # Map emotions to music moods
    emotion_to_mood = {
        'happy': ['upbeat', 'happy', 'cheerful', 'joyful', 'energetic'],
        'sad': ['melancholic', 'sad', 'emotional', 'heartbreak', 'soulful'],
        'angry': ['intense', 'angry', 'powerful', 'energetic', 'rage'],
        'disgust': ['dark', 'intense', 'rebellious', 'alternative'],
        'fear': ['atmospheric', 'suspenseful', 'dramatic', 'haunting'],
        'surprise': ['exciting', 'surprising', 'dynamic', 'uplifting'],
        'neutral': ['relaxing', 'ambient', 'chill', 'smooth', 'indie']
    }
    
    queries = []
    
    # If face detected, focus on emotions
    if face_analysis.get('face_detected', True):
        # Generate queries focused on emotions
        moods = emotion_to_mood.get(mood, emotion_to_mood['neutral'])
        
        # Create mood-focused queries
        for mood_term in moods[:3]:  # Limit to 3 mood terms
            queries.append(f"{mood_term} songs")
            
        # Add more specific queries
        queries.append(f"songs for {mood} mood")
    
    # If no face detected, focus on objects and context
    else:
        objects = detect_objects_from_caption(caption)
        
        # Create object-focused queries
        if objects:
            # Use the most important objects for queries
            main_objects = objects[:3]
            for obj in main_objects:
                queries.append(f"songs about {obj}")
            
            # Combine objects with mood
            queries.append(f"{mood} songs about {' '.join(main_objects[:2])}")
            
            # Broader context query
            queries.append(f"songs inspired by {caption[:30]}")
        else:
            # Fallback to mood-based if no clear objects
            moods = emotion_to_mood.get(mood, emotion_to_mood['neutral'])
            for mood_term in moods[:3]:
                queries.append(f"{mood_term} songs")
    
    # Ensure we have enough unique queries
    queries = list(set(queries))[:5]  # Limit to 5 unique queries
    
    return queries, mood

def parse_iso_duration(iso_duration):
    """Convert ISO 8601 duration format to seconds"""
    # Basic implementation for PT#H#M#S format
    hours = 0
    minutes = 0
    seconds = 0
    
    # Hours
    hour_match = re.search(r'(\d+)H', iso_duration)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Minutes
    minute_match = re.search(r'(\d+)M', iso_duration)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    # Seconds
    second_match = re.search(r'(\d+)S', iso_duration)
    if second_match:
        seconds = int(second_match.group(1))
    
    return hours * 3600 + minutes * 60 + seconds

def search_songs(queries, language_filter='English', max_results=15, user_filters=None, current_video_ids=None):
    """Search for songs using the generated queries with YouTube Data API"""
    all_songs = []
    current_video_ids = current_video_ids or set()
    
    # Use each query to search for songs
    for query in queries:
        try:
            # Add language specifics to the query if needed
            if language_filter == 'Hindi':
                search_query = f"{query} hindi song"
            elif language_filter == 'English':
                search_query = f"{query} english song"
            else:
                search_query = f"{query} song"
                
            # Append genre filters if selected
            if user_filters:
                genre_filters = []
                for filter_name in user_filters:
                    if filter_name in ['Hip-hop', 'Pop']:
                        genre_filters.append(filter_name.lower())
                
                if genre_filters:
                    search_query += " " + " ".join(genre_filters)
            
            # Avoid instrumentals by explicitly requesting songs with lyrics
            search_query += " lyrics music video"
            
            # YouTube Data API search request
            search_response = youtube.search().list(
                q=search_query,
                part="snippet",
                maxResults=20,  # Request more to have enough after filtering
                type="video",
                videoCategoryId="10",  # Music category
                videoEmbeddable="true"
            ).execute()
            
            # Get video IDs for detailed info
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if video_ids:
                # Get detailed video information
                videos_response = youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=','.join(video_ids)
                ).execute()
                
                # Process each video result
                for item in videos_response.get('items', []):
                    # Extract duration in ISO 8601 format
                    iso_duration = item['contentDetails'].get('duration', 'PT0M0S')
                    # Convert ISO 8601 duration to seconds
                    duration_seconds = parse_iso_duration(iso_duration)
                    
                    # Format duration as MM:SS
                    minutes = duration_seconds // 60
                    seconds = duration_seconds % 60
                    duration_str = f"{minutes}:{seconds:02d}"
                    
                    # Filter out songs shorter than 1.5 minutes (90 seconds) or longer than 9 minutes (540 seconds)
                    if 90 <= duration_seconds <= 540:
                        # Skip if this video is already in the current results
                        if item['id'] in current_video_ids:
                            continue
                            
                        # Extract view count
                        view_count = item['statistics'].get('viewCount', '0')
                        formatted_views = f"{int(view_count):,} views"
                        
                        # Get thumbnail
                        thumbnails = item['snippet'].get('thumbnails', {})
                        thumbnail_url = thumbnails.get('high', {}).get('url', '')
                        
                        # Create song entry
                        song = {
                            'title': item['snippet'].get('title', 'Unknown'),
                            'artists': [item['snippet'].get('channelTitle', 'Unknown')],
                            'album': 'YouTube',  # YouTube API doesn't provide album info
                            'duration': duration_str,
                            'durationSeconds': duration_seconds,
                            'thumbnail': thumbnail_url,
                            'videoId': item['id'],
                            'query': query,
                            'viewCount': formatted_views,
                            'rawViewCount': int(view_count)
                        }
                        all_songs.append(song)
        except Exception as e:
            st.error(f"Error searching with query '{query}': {str(e)}")
    
    # Remove duplicates based on videoId
    unique_songs = []
    seen_ids = set()
    for song in all_songs:
        if song['videoId'] not in seen_ids:
            unique_songs.append(song)
            seen_ids.add(song['videoId'])
    
    # Apply additional filters if needed
    filtered_songs = unique_songs
    
    if user_filters:
        if 'Emerging artists' in user_filters:
            # Sort by view count ascending (fewer views first)
            filtered_songs = sorted(filtered_songs, key=lambda x: x.get('rawViewCount', 0))
            
        if 'Popular' in user_filters:
            # Sort by view count descending (more views first)
            filtered_songs = sorted(filtered_songs, key=lambda x: x.get('rawViewCount', 0), reverse=True)
            
        if 'Trending' in user_filters:
            # We don't have actual trending data, so we'll use view count as an approximation
            filtered_songs = sorted(filtered_songs, key=lambda x: x.get('rawViewCount', 0), reverse=True)
    
    return filtered_songs[:max_results]

def display_songs(songs, start_idx=0, batch_size=5):
    """Display songs in batches"""
    end_idx = min(start_idx + batch_size, len(songs))
    current_batch = songs[start_idx:end_idx]
    
    if not current_batch:
        st.write("No more songs found matching the criteria.")
        return False
    
    for song in current_batch:
        cols = st.columns([1, 3])
        
        with cols[0]:
            st.image(song['thumbnail'], width=200)
            
        with cols[1]:
            st.markdown(f"### {song['title']}")
            st.write(f"**Channel/Artist:** {', '.join(song['artists'])}")
            if song['album'] != 'YouTube':
                st.write(f"**Album:** {song['album']}")
            st.write(f"**Duration:** {song['duration']}")
            st.write(f"**Views:** {song.get('viewCount', 'N/A')}")
            
            video_id = song['videoId']
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Add embed option - use session state to track which videos to display
            video_key = f"play_{video_id}"
            
            # Initialize video state if not present
            if video_key not in st.session_state:
                st.session_state[video_key] = False
                
            # Button to toggle video state
            if st.button(f"Play '{song['title']}'", key=f"button_{video_id}"):
                st.session_state[video_key] = not st.session_state[video_key]
                
            # Display video if state is True
            if st.session_state[video_key]:
                st.video(youtube_url)
        
        st.divider()
    
    return True

# Main app
def main():
    st.title("🎧 SyncPixel 📸")
    st.write("Upload an image and get music recommendations based on the emotions and objects detected!")
    
    # Initialize session state for tracking displayed songs
    if 'displayed_songs' not in st.session_state:
        st.session_state.displayed_songs = []
    
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = 0
    
    if 'display_more_button' not in st.session_state:
        st.session_state.display_more_button = False
    
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    
    if 'detected_mood' not in st.session_state:
        st.session_state.detected_mood = ""
    
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
        
        # Add background image to this section
        # Path to your background image (will be replaced by the user)
        bg_image_path = "path/to/your/image.jpg"  # User will replace this path
        
        # Try to load and use background image
        try:
            with open(bg_image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode()
                
            # Apply background image to the about section container with CSS
            st.markdown(
                f"""
                <style>
                .about-section {{
                    background-image: url("data:image/png;base64,{encoded_img}");
                    background-size: cover;
                    background-position: center;
                    color: white;
                    text-shadow: 1px 1px 2px black;
                    padding: 20px;
                    border-radius: 10px;
                }}
                </style>
                """, 
                unsafe_allow_html=True
            )
            bg_class = "about-section"
        except:
            # If image loading fails, use default styling
            bg_class = ""
        
        # About section content with transparent background
        st.markdown(
            f"""
            <div class="{bg_class}">
            <h3>SyncPixel: Music from Images</h3>
            <p>SyncPixel analyzes your uploaded images to recommend music that matches the mood and emotions.</p>
            
            <h4>Features:</h4>
            <ul>
                <li>Analyzes facial expressions to detect emotions</li>
                <li>Generates image captions to understand context</li>
                <li>Recommends music based on detected emotions</li>
                <li>Customizable language and genre preferences</li>
            </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Function to load more songs
    def load_more_songs():
        st.session_state.current_batch += 5
        
        # Check if we need to fetch more songs
        if len(st.session_state.displayed_songs) < st.session_state.current_batch + 5:
            # Get IDs of songs already displayed
            current_video_ids = {song['videoId'] for song in st.session_state.displayed_songs}
            
            # Fetch more songs
            more_songs = search_songs(
                st.session_state.queries, 
                language_option, 
                max_results=10,  # Fetch 10 more
                user_filters=filter_options,
                current_video_ids=current_video_ids
            )
            
            # Add to displayed songs
            st.session_state.displayed_songs.extend(more_songs)
    
    if uploaded_file is not None:
        # Display image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image with a progress bar
        with st.spinner("Analyzing image and finding music..."):
            progress_bar = st.progress(0)
            
            # Step 1: Generate caption
            progress_bar.progress(20)
            caption = get_image_caption(image)
            
            # Step 2: Analyze face for emotions
            progress_bar.progress(40)
            face_analysis = analyze_mood_from_face(image)
            
            # Combine all analysis results
            analysis_results = {
                'caption': caption,
                'face_analysis': face_analysis
            }
            
            # Step 3: Generate search queries
            progress_bar.progress(60)
            queries, detected_mood = generate_search_queries(analysis_results)
            
            # Store queries and mood in session state for "load more" functionality
            st.session_state.queries = queries
            st.session_state.detected_mood = detected_mood
            
            # Step 4: Search for songs - limit to 5 initial songs
            progress_bar.progress(80)
            initial_songs = search_songs(queries, language_option, max_results=10, user_filters=filter_options)
            
            # Store all songs in session state
            st.session_state.displayed_songs = initial_songs
            st.session_state.current_batch = 0
            
            progress_bar.progress(100)
            time.sleep(0.5)  # Small pause to show completion
            progress_bar.empty()  # Remove progress bar
        
        # Display analysis results
        with col2:
            st.subheader("Image Analysis")
            
            # Display caption with larger text
            st.markdown(f"<h3 style='font-size: 24px;'>Caption: {caption}</h3>", unsafe_allow_html=True)
            
            st.write("**Emotion Analysis:**")
            if face_analysis['face_detected']:
                emotion_data = face_analysis['emotions']
                
                # Create bar chart for emotions instead of pie chart
                fig, ax = plt.subplots(figsize=(8, 5))
                emotions = list(emotion_data.keys())
                values = list(emotion_data.values())
                
                # Define colors for different emotions
                colors = {
                    'happy': '#FFD700',    # Gold
                    'sad': '#4169E1',      # Royal Blue
                    'angry': '#DC143C',    # Crimson
                    'fear': '#800080',     # Purple
                    'disgust': '#006400',  # Dark Green
                    'surprise': '#FF8C00', # Dark Orange
                    'neutral': '#808080'   # Gray
                }
                
                # Create bar colors list
                bar_colors = [colors.get(emotion, '#808080') for emotion in emotions]
                
                # Create horizontal bar chart (easier to read emotion labels)
                bars = ax.barh(emotions, values, color=bar_colors)
                
                # Add percentage labels on the bars
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width + 1  # Adjust for label positioning
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                            va='center', fontsize=10)
                
                # Add labels and title
                ax.set_xlabel('Probability %')
                ax.set_title('Detected Emotions')
                
                # Adjust layout
                plt.tight_layout()
                
                # Display the chart
                st.pyplot(fig)
                
                # Display dominant emotion
                st.metric("Dominant Emotion", face_analysis['dominant_emotion'].capitalize())
            else:
                st.info("No faces detected. Recommending songs based on objects and context from the image.")
                st.metric("Detected Emotion", detected_mood.capitalize())
                
                # Display detected objects from caption
                objects = detect_objects_from_caption(caption)
                st.write("**Key elements detected:**", ", ".join(objects[:5]).capitalize())
        
        # Display music recommendations
        st.subheader("Recommended Songs")
        
        if len(st.session_state.displayed_songs) == 0:
            st.warning("No songs could be found based on the image analysis. Try another image or different language settings.")
        else:
            # Display the first batch of songs
            display_songs(st.session_state.displayed_songs, st.session_state.current_batch, 5)
            
            # Add "View 5 more songs" button if there are more songs to display
            more_songs_available = len(st.session_state.displayed_songs) > st.session_state.current_batch + 5
            
            if more_songs_available or st.session_state.queries:
                if st.button("View 5 more songs", key=f"load_more_{st.session_state.current_batch}"):
                    load_more_songs()
                    st.rerun()

if __name__ == "__main__":
    main()
