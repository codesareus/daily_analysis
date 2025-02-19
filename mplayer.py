import base64
import streamlit as st

# Assuming 'music' is a list of file paths to your audio files
music = ['1.mp3', '2.mp3', '3.mp3', '4.mp3'] # Replace with your actual audio file path(s)

def play_music(number=0):
    # Open the audio file and encode it in base64
    audio_file = open(music[number], "rb")
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Create the HTML for the audio player with autoplay
    audio_html = f"""
        <audio id="audioPlayer" controls>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
    """
    
    # Add JavaScript to stop the music after 10 seconds and restart when a button is clicked
    javascript_code = """
        <script>
            // Function to stop the audio
            function stopAudio() {
                var audio = document.getElementById('audioPlayer');
                if (audio) {{
                    audio.pause();
                    audio.currentTime = 0;  // Rewind the audio to the start
                }}
            }

            // Function to play the audio
            function playAudio() {
                var audio = document.getElementById('audioPlayer');
                if (audio) {{
                    audio.play();
                }}
            }

            // Automatically stop the music after 10 seconds (10000 milliseconds)
            setTimeout(stopAudio, 10000);
        </script>
    """
    
    # Display the audio player and the JavaScript code
    st.markdown(audio_html, unsafe_allow_html=True)
    st.markdown(javascript_code, unsafe_allow_html=True)

    # Add a "Restart Music" button
    if st.button("Restart Music"):
        st.markdown("""
            <script>
                playAudio();
            </script>
        """, unsafe_allow_html=True)

# Call the function to play music
