import threading
import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import tempfile
from gtts import gTTS
import os
import json
import requests
import toml
from google.cloud import speech
from openai import OpenAI

# Load secrets from .streamlit/secrets.toml
with open('.streamlit/secrets.toml', 'r') as f:
    secrets = toml.load(f)

# Create a temporary file with the Google Cloud credentials
google_cloud_credentials = secrets["gcp_service_account"]

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
    json.dump(google_cloud_credentials, temp_file)
    temp_file_path = temp_file.name

# Set the environment variable to the path of the temporary file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
client_openai = OpenAI()

def check_network():
    try:
        response = requests.get("https://speech.googleapis.com", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

if not check_network():
    st.error("Unable to connect to Google Speech-to-Text API. Please check your network connection.")

# List available audio devices
device_list = sd.query_devices()
device_names = [device['name'] for device in device_list]

# Streamlit UI setup
st.title("Real-Time Speech-Enabled Chat with GPT")
st.write("Upload your resume and provide some context to start the chat (optional).")

# Select the audio input device (built-in microphone, Loopback virtual device, etc.)
selected_device = st.selectbox("Select Input Device", device_names)
input_device_index = device_names.index(selected_device)

# Upload resume file
uploaded_file = st.file_uploader("Upload your resume (optional)", type=["pdf", "docx", "txt"])

# Input context
context_input = st.text_area("Context (optional, e.g., job description, resume details)", height=150)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    context_input += f"\nResume uploaded: {uploaded_file.name}"

# Debug placeholder
debug_placeholder = st.empty()

# Audio queue to communicate with the callback
audio_queue = queue.Queue()

class ChatSystem:
    def __init__(self):
        self.context = context_input if context_input else "No context provided."
        self.chat_history = [{"role": "system", "content": self.context}]
        self.stop_signal = False
        self.debug_info = []

    def set_context(self, new_context):
        self.context = new_context
        self.chat_history.append({"role": "system", "content": self.context})

    def start_listening(self):
        self.stop_signal = False
        threading.Thread(target=self.listen_and_transcribe, daemon=True).start()

    def stop_listening(self):
        self.stop_signal = True

    def update_debug_info(self):
        """Update the Streamlit app with the latest debug info."""
        debug_placeholder.text("\n".join(self.debug_info))

    def audio_callback(self, indata, frames, time, status):
        """Process and enqueue audio data in real-time."""
        if status:
            self.debug_info.append(f"Audio callback status: {status}")
        self.debug_info.append(f"Audio callback received data: {len(indata)} bytes")

        # Calculate RMS to determine if there's significant sound
        rms = np.sqrt(np.mean(indata ** 2))
        self.debug_info.append(f"RMS value: {rms}")

        if rms > 0.01:  # Threshold for detecting sound
            audio_queue.put(indata.copy())
            self.debug_info.append(f"Enqueued {len(indata)} bytes of audio data")
        else:
            self.debug_info.append("Silence detected, not sending to queue.")

        self.update_debug_info()

    def listen_and_transcribe(self):
        client_speech = speech.SpeechClient()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=False
        )

        def audio_generator():
            while not self.stop_signal:
                try:
                    chunk = audio_queue.get(block=True, timeout=0.1)
                    self.debug_info.append(f"Dequeued {len(chunk)} bytes of audio data")
                    audio_data = (chunk * 32767).astype(np.int16).tobytes()
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                except queue.Empty:
                    self.debug_info.append("Queue empty")
                    continue

        with sd.InputStream(samplerate=44100, channels=1, callback=self.audio_callback, device=input_device_index, latency='low', blocksize=1024):
            try:
                requests = audio_generator()
                responses = client_speech.streaming_recognize(streaming_config, requests)

                for response in responses:
                    if self.stop_signal:
                        break
                    for result in response.results:
                        if result.is_final:
                            transcribed_text = result.alternatives[0].transcript
                            self.debug_info.append(f"Transcribed: {transcribed_text}")
                            st.write(f"You said: {transcribed_text}")
                            response_text = self.ask_gpt(transcribed_text)
                            self.debug_info.append(f"ChatGPT response: {response_text}")
                            self.chat_history.append({"role": "assistant", "content": response_text})
                            st.write(f"ChatGPT says: {response_text}")
                            self.speak_text(response_text)
                    self.update_debug_info()

            except Exception as e:
                self.debug_info.append(f"Error during transcription: {str(e)}")
                st.write(f"Error during transcription: {str(e)}")
                self.update_debug_info()

    def ask_gpt(self, query):
        try:
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.chat_history,
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.debug_info.append(f"Error in GPT response: {str(e)}")
            st.write(f"Error in GPT response: {str(e)}")
            self.update_debug_info()
            return "Sorry, I couldn't generate a response at this time."

    @staticmethod
    def speak_text(text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3_file:
            tts.save(temp_mp3_file.name)
            temp_mp3_file_path = temp_mp3_file.name

        if os.name == 'posix':  # macOS/Linux
            os.system(f"afplay {temp_mp3_file_path}")
        elif os.name == 'nt':  # Windows
            os.system(f"start {temp_mp3_file_path}")

# Initialize the chat system
chat_system = ChatSystem()

# Set the context in the chat system
if st.button("Set Context"):
    chat_system.set_context(context_input)
    st.write("Context set successfully.")

# Display chat history
st.subheader("Chat History")
for message in chat_system.chat_history:
    st.write(f"{message['role'].capitalize()}: {message['content']}")

# Create a placeholder for the listening status
listening_status = st.empty()

# Button to toggle speech recognition
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

if st.button("Toggle Voice Recognition"):
    st.session_state.is_listening = not st.session_state.is_listening
    if st.session_state.is_listening:
        listening_status.text("Listening... Click 'Toggle Voice Recognition' to stop.")
        chat_system.start_listening()
    else:
        listening_status.text("Voice recognition stopped.")
        chat_system.stop_listening()

# Show the debug information continuously
st.subheader("Debug Information")
chat_system.update_debug_info()

# Add an audio input test
if st.button("Test Audio Input"):
    st.write("Recording 5 seconds of audio...")
    duration = 5  # seconds
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, device=input_device_index)
    sd.wait()
    st.write("Recording complete. Analyzing...")

    # Calculate the RMS of the recording to check if there's audio
    rms = np.sqrt(np.mean(recording ** 2))
    st.write(f"Audio RMS: {rms}")

    if rms > 0.01:  # You may need to adjust this threshold
        st.write("Audio input is working. Voice detected.")
    else:
        st.write("No significant audio detected. Please check your microphone.")
