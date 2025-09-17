import streamlit as st
import tempfile
import os
from moviepy.editor import VideoFileClip
import numpy as np
import soundfile as sf
import noisereduce as nr
import whisper
import pandas as pd
import io

# Set page config
st.set_page_config(page_title="Classroom Interaction Analysis", layout="wide", page_icon="🎥")

# Initialize session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'extracted_audio' not in st.session_state:
    st.session_state.extracted_audio = None
if 'cleaned_audio' not in st.session_state:
    st.session_state.cleaned_audio = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video using MoviePy"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        audio.close()
        video.close()
        return True
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return False

def remove_noise_from_audio(input_audio_path, output_audio_path):
    """Remove noise using noisereduce library"""
    try:
        # Load audio file
        data, samplerate = sf.read(input_audio_path)
        
        # Reduce noise
        reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
        
        # Save cleaned audio
        sf.write(output_audio_path, reduced_noise, samplerate)
        return True
    except Exception as e:
        st.error(f"Error removing noise: {str(e)}")
        return False

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

# App Header
st.title("🎥 Classroom Interaction Analysis")
st.write("Upload a classroom video to extract audio, remove noise, and generate transcripts")

# Sidebar for navigation
with st.sidebar:
    st.header("📋 Process Steps")
    st.info("""
    1. Upload video file
    2. Extract audio from video
    3. Remove background noise
    4. Generate transcript
    5. Download results
    """)
    
    st.header("📄 Supported Formats")
    st.write("*Video:* MP4, AVI, MOV, MKV")
    st.write("*Audio:* WAV, MP3")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Step 1: Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a classroom video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload your classroom recording for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.processed_video = video_path
        
        # Display video
        st.video(uploaded_file)
        st.success("✅ Video uploaded successfully!")

with col2:
    st.subheader("🔊 Step 2: Extract Audio")
    
    if st.session_state.processed_video:
        if st.button("🎵 Extract Audio from Video", type="primary"):
            with st.spinner("Extracting audio..."):
                # Create temporary file for audio
                audio_path = st.session_state.processed_video.replace('.mp4', '.wav')
                
                if extract_audio_from_video(st.session_state.processed_video, audio_path):
                    st.session_state.extracted_audio = audio_path
                    st.success("✅ Audio extracted successfully!")
                    
                    # Display audio player
                    with open(audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
    else:
        st.info("Upload a video file first")

# Noise removal section
st.subheader("🎚 Step 3: Remove Background Noise")

col3, col4 = st.columns([1, 1])

with col3:
    if st.session_state.extracted_audio:
        if st.button("🔇 Remove Background Noise", type="primary"):
            with st.spinner("Removing noise... This may take a moment"):
                cleaned_audio_path = st.session_state.extracted_audio.replace('.wav', '_clean.wav')
                
                if remove_noise_from_audio(st.session_state.extracted_audio, cleaned_audio_path):
                    st.session_state.cleaned_audio = cleaned_audio_path
                    st.success("✅ Noise removed successfully!")
    else:
        st.info("Extract audio first")

with col4:
    if st.session_state.cleaned_audio:
        st.write("🎧 Cleaned Audio:")
        with open(st.session_state.cleaned_audio, 'rb') as audio_file:
            cleaned_audio_bytes = audio_file.read()
        st.audio(cleaned_audio_bytes, format='audio/wav')
        
        # Download cleaned audio
        st.download_button(
            label="💾 Download Clean Audio",
            data=cleaned_audio_bytes,
            file_name="cleaned_audio.wav",
            mime="audio/wav"
        )

# Transcription section
st.subheader("📝 Step 4: Generate Transcript")

col5, col6 = st.columns([1, 1])

with col5:
    audio_for_transcription = st.session_state.cleaned_audio or st.session_state.extracted_audio
    
    if audio_for_transcription:
        if st.button("📄 Generate Transcript", type="primary"):
            with st.spinner("Transcribing audio... This may take several minutes"):
                transcript = transcribe_audio(audio_for_transcription)
                if transcript:
                    st.session_state.transcript = transcript
                    st.success("✅ Transcript generated successfully!")
    else:
        st.info("Extract audio first")

with col6:
    if st.session_state.transcript:
        st.text_area(
            "Generated Transcript:",
            st.session_state.transcript,
            height=200,
            help="Automatically generated transcript from your audio"
        )
        
        # Create CSV data
        transcript_data = pd.DataFrame({
            'Timestamp': ['Full Recording'],
            'Speaker': ['Unknown'],
            'Text': [st.session_state.transcript]
        })
        
        csv_data = transcript_data.to_csv(index=False)
        
        # Download transcript
        st.download_button(
            label="💾 Download Transcript (CSV)",
            data=csv_data,
            file_name="classroom_transcript.csv",
            mime="text/csv"
        )
        
        # Download as text file
        st.download_button(
            label="💾 Download Transcript (TXT)",
            data=st.session_state.transcript,
            file_name="classroom_transcript.txt",
            mime="text/plain"
        )

# Progress summary
st.subheader("📊 Process Summary")

progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)

with progress_col1:
    if st.session_state.processed_video:
        st.success("✅ Video Uploaded")
    else:
        st.info("⏳ Upload Video")

with progress_col2:
    if st.session_state.extracted_audio:
        st.success("✅ Audio Extracted")
    else:
        st.info("⏳ Extract Audio")

with progress_col3:
    if st.session_state.cleaned_audio:
        st.success("✅ Noise Removed")
    else:
        st.info("⏳ Remove Noise")

with progress_col4:
    if st.session_state.transcript:
        st.success("✅ Transcript Ready")
    else:
        st.info("⏳ Generate Transcript")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>Made by Ritu Soni | Classroom Interaction Analysis Tool</p>
        <p><em>Process your classroom recordings efficiently with AI-powered transcription</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Cleanup temporary files when app restarts
if st.button("🗑 Clear All Data", help="Clean up temporary files and restart"):
    # Clean up temporary files
    for file_path in [st.session_state.processed_video, st.session_state.extracted_audio, st.session_state.cleaned_audio]:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
    
    # Reset session state
    st.session_state.processed_video = None
    st.session_state.extracted_audio = None
    st.session_state.cleaned_audio = None
    st.session_state.transcript = ""
    
    st.success("All data cleared! Refresh the page to start over.")