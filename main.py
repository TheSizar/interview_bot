import openai
import streamlit as st
import speech_recognition as sr
import pyttsx3
import tempfile

# Set up OpenAI API key
openai.api_key = st.secrets["OPEN_AI_API"]

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()


# Define a function to handle the context and chat
class ChatSystem:
    def __init__(self):
        self.context = ""

    def set_context(self, context):
        self.context = context

    def listen_and_respond(self):
        # Use the system's default microphone as the audio source
        with sr.Microphone() as source:
            st.write("Listening for speech...")
            audio = recognizer.listen(source)

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                st.write(f"You said: {text}")

                # Send the text to OpenAI API if it's a question
                response = self.ask_gpt(text)
                st.write(f"ChatGPT says: {response}")

                # Convert response text to speech
                tts_engine.say(response)
                tts_engine.runAndWait()

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


# Initialize the chat system
chat_system = ChatSystem()

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

# Set the context in the chat system
if st.button("Set Context"):
    chat_system.set_context(context)
    st.write("Context set successfully.")

# Button to activate speech recognition
if st.button("Activate Speech Recognition"):
    chat_system.listen_and_respond()

