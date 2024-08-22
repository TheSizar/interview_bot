import openai
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from gtts import gTTS
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
client = OpenAI()

# List available audio devices
device_list = sd.query_devices()
device_names = [device['name'] for device in device_list]

# Streamlit UI setup
st.title("Speech-Enabled Chat with GPT")
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

# Define a function to handle the context and chat
class ChatSystem:
    def __init__(self):
        self.context = ""
        self.chat_history = []

    def set_context(self, new_context):
        self.context = new_context
        self.chat_history.append({"role": "system", "content": self.context})

    @staticmethod
    def record_audio(duration=5, fs=44100):
        """Record audio from the selected microphone."""
        st.write("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=input_device_index)
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

        # Simulate transcribing text from the audio (placeholder)
        # In a real implementation, you would use a speech-to-text service here
        transcribed_text = "Simulated transcribed text from the recorded audio"

        st.write(f"You said: {transcribed_text}")
        self.chat_history.append({"role": "user", "content": transcribed_text})

        # Send the text to OpenAI API
        response = self.ask_gpt(transcribed_text)
        st.write(f"ChatGPT says: {response}")
        self.chat_history.append({"role": "assistant", "content": response})

        # Convert response text to speech using gTTS
        self.speak_text(response)

        # You can delete the audio file after processing, if you don't need it anymore
        os.remove(audio_file_path)

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
if st.button("Activate Speech Recognition"):
    chat_system.listen_and_respond()