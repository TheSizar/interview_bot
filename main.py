import openai
import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from scipy.io import wavfile
import os
from gtts import gTTS

# Set up OpenAI API key
openai.api_key = st.secrets["OPEN_AI_API"]

# Streamlit UI setup
st.title("Speech-Enabled Chat with GPT")
st.write("Upload your resume and provide some context to start the chat.")

# Upload resume file
uploaded_file = st.file_uploader("Upload your resume (optional)", type=["pdf", "docx", "txt"])

# Input context
context = st.text_area("Context (e.g., job description, resume details)", height=150)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    context += f"\nResume uploaded: {uploaded_file.name}"


# Define a function to handle the context and chat
class ChatSystem:
    def __init__(self):
        self.context = ""

    def set_context(self, context):
        self.context = context

    def record_audio(self, duration=5, fs=44100):
        """Record audio from the microphone."""
        st.write("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        st.write("Recording finished")

        # Save recording to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            write(temp_wav_file.name, fs, recording)
            temp_wav_file_path = temp_wav_file.name

        return temp_wav_file_path

    def listen_and_respond(self, duration=5):
        """Record audio, transcribe it, and generate a response using GPT."""
        audio_file_path = self.record_audio(duration=duration)

        # Read the audio file
        fs, audio = wavfile.read(audio_file_path)

        # Here you would typically use a speech recognition model to convert the audio to text
        # For the sake of this example, let's assume we have the transcribed text
        text = "Simulated transcribed text from the recorded audio"

        st.write(f"You said: {text}")

        # Send the text to OpenAI API if it's a question
        response = self.ask_gpt(text)
        st.write(f"ChatGPT says: {response}")

        # Convert response text to speech using gTTS
        self.speak_text(response)

    def ask_gpt(self, query):
        # Combine the context with the query
        prompt = f"Context: {self.context}\nUser: {query}\nAI:"

        # Call the OpenAI API with the prompt
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def speak_text(self, text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3_file:
            tts.save(temp_mp3_file.name)
            temp_mp3_file_path = temp_mp3_file.name

        # Use 'afplay' on macOS, 'start' on Windows
        if os.name == 'posix':  # macOS/Linux
            os.system(f"afplay {temp_mp3_file_path}")
        elif os.name == 'nt':  # Windows
            os.system(f"start {temp_mp3_file_path}")


# Initialize the chat system
chat_system = ChatSystem()

# Set the context in the chat system
if st.button("Set Context"):
    chat_system.set_context(context)
    st.write("Context set successfully.")

# Button to activate speech recognition
if st.button("Activate Speech Recognition"):
    chat_system.listen_and_respond()
