import openai
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile

# Set up OpenAI API key
openai.api_key = st.secrets["OPEN_AI_API"]

# Initialize the recognizer
recognizer = sr.Recognizer()

# List all available microphones
mic_list = sr.Microphone.list_microphone_names()

# Streamlit UI setup
st.title("Speech-Enabled Chat with GPT")
st.write("Upload your resume and provide some context to start the chat.")

# Display available input devices
st.write("Select the input device (microphone):")
selected_mic = st.selectbox("Choose a microphone or virtual device", mic_list)

# Function to get the selected microphone's device index
def get_device_index(mic_name):
    for index, name in enumerate(mic_list):
        if name == mic_name:
            return index
    return None

# Get the selected microphone's device index
device_index = get_device_index(selected_mic)

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

    def listen_and_respond(self, device_index):
        # Use the selected microphone or virtual device as the audio source
        with sr.Microphone(device_index=device_index) as source:
            st.write("Listening for speech...")
            audio = recognizer.listen(source)

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                st.write(f"You said: {text}")

                # Send the text to OpenAI API if it's a question
                response = self.ask_gpt(text)
                st.write(f"ChatGPT says: {response}")

                # Convert response text to speech using gTTS
                self.speak_text(response)

            except Exception as e:
                st.write(f"Error: {e}")

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
        tts.save("output.mp3")
        # Use 'afplay' on macOS, 'start' on Windows
        if os.name == 'posix':  # macOS/Linux
            os.system("afplay output.mp3")
        elif os.name == 'nt':  # Windows
            os.system("start output.mp3")

# Initialize the chat system
chat_system = ChatSystem()

# Set the context in the chat system
if st.button("Set Context"):
    chat_system.set_context(context)
    st.write("Context set successfully.")

# Button to activate speech recognition
if st.button("Activate Speech Recognition"):
    chat_system.listen_and_respond(device_index)
