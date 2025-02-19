import base64
import streamlit as st

def play_music(number=0):
    # Assuming 'music' is a list of file paths to your audio files
    audio_file = open(music[number], "rb")
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Remove the 'muted' attribute to allow sound
    audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
