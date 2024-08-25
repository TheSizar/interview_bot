
import streamlit as st
import sounddevice as sd
import queue
import tempfile
from gtts import gTTS
import os
import sys  # Import sys module
from openai import OpenAI
from google.cloud import speech

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
client = OpenAI()

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = st.secrets["GOOGLE_CLOUD_API_KEY"]

# List available audio devices
device_list = sd.query_devices()
device_names = [device['name'] for device in device_list]

# Streamlit UI setup
st.title("Real-Time Speech-Enabled Chat with GPT")
st.write("Upload your resume and provide some context to start the chat.")

# Select the audio input device (built-in microphone, Loopback virtual device, etc.)
selected_device = st.selectbox("Select Input Device", device_names)

# Set the input device
input_device_index = device_names.index(selected_device)

# Upload resume file
uploaded_file = st.file_uploader("Upload your resume (optional)", type=["pdf", "docx", "txt"])

# Input context
context_input = st.text_area("Context (e.g., job description, resume details)", height=150)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    context_input += f"\nResume uploaded: {uploaded_file.name}"

# Audio queue to communicate with the callback
audio_queue = queue.Queue()

# Define a function to handle the context and chat


class ChatSystem:
    def __init__(self):
        self.context = ""
        self.chat_history = []

    def set_context(self, new_context):
        self.context = new_context
        self.chat_history.append({"role": "system", "content": self.context})

    @staticmethod  # Declare as static since 'self' is not used
    def audio_callback(indata, status):
        """Callback function to stream audio data to the Google API."""
        if status:
            st.write(status, file=sys.stderr)
        audio_queue.put(bytes(indata))

    def listen_and_transcribe(self):
        """Capture and transcribe audio in real-time."""
        client = speech.SpeechClient()
        # ure the Google Speech API stream
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True  # Allows partial (real-time) transcription
        )

        # Start streaming the audio input
        with sd.InputStream(samplerate=44100, channels=1, callback=self.audio_callback, device=input_device_index):
            audio_generator = self.audio_stream_generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests)

            # Process the responses as they come in
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        transcribed_text = result.alternatives[0].transcript
                        st.write(f"You said: {transcribed_text}")
                        self.chat_history.append({"role": "user", "content": transcribed_text})

                        # Get GPT response and output
                        response_text = self.ask_gpt(transcribed_text)
                        st.write(f"ChatGPT says: {response_text}")
                        self.chat_history.append({"role": "assistant", "content": response_text})
                        self.speak_text(response_text)

    def audio_stream_generator(self):
        """Generator to yield audio data to Google API."""
        while True:
            data = audio_queue.get()
            if data is None:
                break
            yield data

    def ask_gpt(self, query):
        # Use the correct method to call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.chat_history,
            max_tokens=300,  # Adjust response length
            temperature=0.3  # Adjust creativity (0.0 - precise, 1.0 - creative)
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def speak_text(text):
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
    chat_system.set_context(context_input)
    st.write("Context set successfully.")

# Display chat history
st.subheader("Chat History")
for message in chat_system.chat_history:
    st.write(f"{message['role'].capitalize()}: {message['content']}")

# Button to activate speech recognition
if st.button("Activate Real-Time Speech Recognition"):
    chat_system.listen_and_transcribe()
