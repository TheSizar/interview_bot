import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import queue
import tempfile
import json
import os
import toml
from gtts import gTTS
from google.cloud import speech
import openai
from openai import OpenAI
import docx
import PyPDF2
from tkinter import ttk
from ttkthemes import ThemedTk
from threading import Lock  # Import Lock for thread safety

# Load secrets from the secrets.toml
with open('../secrets.toml', 'r') as f:
    secrets = toml.load(f)

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = secrets["OPEN_AI_API"]

client_openai = OpenAI()

# Create a temporary file with the Google Cloud credentials
google_cloud_credentials = secrets["gcp_service_account"]
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
    json.dump(google_cloud_credentials, temp_file)
    temp_file_path = temp_file.name

# Set the environment variable to the path of the temporary file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

# Initialize Google Speech-to-Text client and queue
client_speech = speech.SpeechClient()
audio_queue = queue.Queue()

class AudioLevelMeter(tk.Canvas):
    def __init__(self, master, device_index, **kwargs):
        super().__init__(master, **kwargs)
        self.device_index = device_index
        self.levels = [0] * 20  # Initial levels
        self.create_rectangles()
        self.update_meter()

    def create_rectangles(self):
        self.bars = []
        for i in range(20):
            bar = self.create_rectangle(i * 15 + 5, 20, i * 15 + 15, 80, fill="gray")
            self.bars.append(bar)

    def update_meter(self):
        # Get the current audio input level
        level = self.get_input_level()
        # Update the level bars based on the current level
        self.update_bars(level)
        # Schedule the next update
        self.after(100, self.update_meter)  # Update every 100ms

    def update_bars(self, level):
        # Normalize level to 0-20 range
        num_active_bars = int(level / 2)  # Adjust this factor to make the meter more sensitive
        for i, bar in enumerate(self.bars):
            if i < num_active_bars:
                self.itemconfig(bar, fill="green")
            else:
                self.itemconfig(bar, fill="gray")

    def get_input_level(self):
        duration = 0.1  # duration to measure level (100ms)
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, device=self.device_index)
        sd.wait()
        rms = np.sqrt(np.mean(recording ** 2))
        return rms * 1000  # Increase the multiplier to make the meter more sensitive


class SpeechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech-Enabled Chat with GPT")
        self.root.geometry("1000x800")
        self.is_waiting_for_response = False
        self.is_speaking = False

        self.context = ""
        self.chat_history = [{"role": "system", "content": "Context not set."}]
        self.stop_signal = False
        self.uploaded_files = []
        self.lock = threading.Lock()  # Initialize the lock for thread safety

        # Initialize the list of input devices
        self.device_list = sd.query_devices()
        self.device_names = [device['name'] for device in self.device_list]

        # GUI Components
        self.setup_gui()

    def setup_gui(self):
        # Configure the root window
        self.root.configure(bg='white')

        # Top frame (API Test & Mic Check)
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        self.test_api_button = ttk.Button(top_frame, text="Test Google & OpenAI APIs", command=self.test_apis)
        self.test_api_button.pack(side=tk.LEFT, padx=10)

        self.device_var = tk.StringVar(self.root)
        self.device_var.set(self.device_names[0])  # Default to the first device
        self.device_menu = ttk.Combobox(top_frame, textvariable=self.device_var, values=self.device_names,
                                        state="readonly", width=30)
        self.device_menu.pack(side=tk.LEFT, padx=10)

        self.audio_level_meter = AudioLevelMeter(top_frame, device_index=self.device_names.index(self.device_var.get()),
                                                 width=310, height=100)
        self.audio_level_meter.pack(side=tk.LEFT, padx=10)

        # Middle frame (Context & File Upload)
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(fill=tk.BOTH, expand=True)

        self.job_label = ttk.Label(middle_frame, text="Job Context:", font=('Arial', 12, 'bold'))
        self.job_label.pack(anchor='w', padx=5)

        self.job_input = scrolledtext.ScrolledText(middle_frame, width=80, height=5, wrap=tk.WORD, font=('Arial', 12))
        self.job_input.pack(fill=tk.X, padx=5, pady=5)

        self.context_label = ttk.Label(middle_frame, text="General Context:", font=('Arial', 12, 'bold'))
        self.context_label.pack(anchor='w', padx=5)

        self.context_input = scrolledtext.ScrolledText(middle_frame, width=80, height=5, wrap=tk.WORD,
                                                       font=('Arial', 12))
        self.context_input.pack(fill=tk.X, padx=5, pady=5)

        # Prepopulate the General Context field with the default context
        default_system_message = (
            "You are an AI assistant designed to help with answering difficult questions during job interviews. "
            "Your primary objectives are:\n"
            "1. **Behavioral Questions (e.g., 'Tell me about a time when...')**:\n"
            "  - When asked a behavioral question, generate a model answer using specific examples.\n"
            "  - Structure your response in bullet points, detailing:\n"
            "    - **The Situation**: Describe the situation or challenge.\n"
            "    - **The Task**: Explain what needed to be done.\n"
            "    - **The Action**: Detail the actions taken.\n"
            "    - **The Result**: Highlight the outcomes or achievements.\n"
            "2. **General and Motivational Questions (e.g., 'Why do you want to join our company?')**:\n"
            "  - Craft precise and compelling responses, integrating relevant information from the job description "
            "    and context provided. Emphasize alignment between the user's experience and the role or company.\n"
            "Always provide thoughtful, detailed, and contextually relevant responses."
        )
        self.context_input.insert(tk.END, default_system_message)

        # File Upload (Button and File List on the Same Line)
        file_frame = ttk.Frame(middle_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        self.upload_button = ttk.Button(file_frame, text="Upload File", command=self.upload_file, width=15)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.file_list = ttk.Treeview(file_frame, columns=("File",), show="headings", height=2)
        self.file_list.heading("File", text="Uploaded Files")
        self.file_list.column("File", width=200)  # Adjust column width as needed
        self.file_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Bottom frame (Chat)
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.start_button = ttk.Button(bottom_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(bottom_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.speaking_indicator = ttk.Label(bottom_frame, text="●", foreground="grey")
        self.speaking_indicator.pack(side=tk.LEFT, padx=5)

        # Transcript Display (iMessage Style)
        self.transcript_display = scrolledtext.ScrolledText(bottom_frame, height=20, width=80, wrap=tk.WORD,
                                                            font=('Arial', 12))
        self.transcript_display.pack(fill=tk.BOTH, expand=True)
        self.transcript_display.tag_configure("user", foreground="white", background="#007AFF", justify="right",
                                              font=('Arial', 12, 'bold'), lmargin1=10, rmargin=10)
        self.transcript_display.tag_configure("assistant", foreground="black", background="#E5E5EA", justify="left",
                                              font=('Arial', 12), lmargin1=10, rmargin=10)

        # Reduce unnecessary idle tasks to improve scrolling performance
        self.root.after_idle(self.root.update_idletasks)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("Word documents", "*.docx"), ("Text files", "*.txt")])
        if not file_path:
            return

        try:
            file_name = os.path.basename(file_path)
            self.uploaded_files.append((file_name, file_path))
            self.file_list.insert("", "end", values=(file_name,))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {str(e)}")

    def test_apis(self):
        try:
            # Test Google Speech-to-Text API with a minimal configuration
            client = speech.SpeechClient()

            # Using a sample audio file from Google Cloud Storage for the test
            audio = speech.RecognitionAudio(uri="gs://cloud-samples-data/speech/brooklyn_bridge.raw")
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )

            response = client.recognize(config=config, audio=audio)

            if response:
                print("Google Speech-to-Text API connection successful.")
            else:
                raise Exception("Failed to get a valid response from Google Speech-to-Text API")

            # Test OpenAI API by creating a simple completion request
            completion = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Test"}],
            )

            if completion:
                print("OpenAI API connection successful.")
            else:
                raise Exception("Failed to get a valid response from OpenAI API")

            messagebox.showinfo("Success", "Successfully connected to Google & OpenAI APIs.")
        except openai.error.OpenAIError as e:
            messagebox.showerror("Error", f"OpenAI API error: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"API Connection Test Failed: {str(e)}")

    def start_recording(self):
        self.stop_signal = False
        self.initial_context = self.prepare_initial_context()
        self.chat_history = []  # Initialize or clear the chat history for user and assistant messages
        threading.Thread(target=self.listen_and_transcribe, daemon=True).start()

    def prepare_initial_context(self):
        context = self.context_input.get("1.0", tk.END).strip()
        job_context = self.job_input.get("1.0", tk.END).strip()

        # Prepare the initial system message with context and job description
        system_message = f"Job Context: {job_context}\nGeneral Context: {context}\n\n"
        system_message += "Uploaded files:\n"

        for file_name, file_path in self.uploaded_files:
            system_message += f"- {file_name}\n"
            file_content = self.read_file_content(file_path)
            system_message += f"Content: {file_content[:500]}...\n\n"  # Include first 500 characters of each file

        return {"role": "system", "content": system_message}

    def read_file_content(self, file_path):
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith(".pdf"):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join([page.extract_text() for page in reader.pages])
        return ""

    def stop_recording(self):
        self.stop_signal = True
        try:
            audio_queue.put(None)  # Stop the audio generator
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping recording: {str(e)}")

    def listen_and_transcribe(self):
        selected_device = self.device_var.get()
        device_index = self.device_names.index(selected_device)

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

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            audio_queue.put(indata.copy())

        def audio_generator():
            while not self.stop_signal:
                try:
                    chunk = audio_queue.get(block=True, timeout=0.1)
                    if chunk is None:  # Check for stop signal
                        break
                    audio_data = (chunk * 32767).astype(np.int16).tobytes()
                    yield speech.StreamingRecognizeRequest(audio_content=audio_data)
                except queue.Empty:
                    continue

        try:
            with sd.InputStream(samplerate=44100, channels=1, callback=audio_callback, device=device_index,
                                latency='low',
                                blocksize=1024):
                requests = audio_generator()
                responses = client_speech.streaming_recognize(streaming_config, requests=requests)

                for response in responses:
                    if self.stop_signal:
                        break
                    for result in response.results:
                        if result.is_final:
                            transcribed_text = result.alternatives[0].transcript
                            self.transcript_display.insert(tk.END, f"You: {transcribed_text}\n", "user")
                            self.transcript_display.see(tk.END)

                            # Only call GPT if we're not already waiting for a response
                            if self.is_question(transcribed_text) and not self.is_waiting_for_response:
                                self.is_waiting_for_response = True  # Indicate that we're waiting for a response
                                threading.Thread(target=self.process_gpt_response, args=(transcribed_text,)).start()

        except Exception as e:
            messagebox.showerror("Error", f"Error during transcription: {str(e)}")

    def is_question(self, text):
        """Determine if the text is a question."""
        question_words = ["who", "what", "where", "when", "why", "how", "is", "are", "do", "does", "can", "could",
                          "would", "should"]
        return text.strip().endswith("?") or text.split()[0].lower() in question_words

    def ask_gpt(self, query):
        try:
            # Add the user's question to the chat history
            self.chat_history.append({"role": "user", "content": query})

            # Prepare the messages for the API call
            messages = [self.initial_context]  # Start with the initial context

            # Append the last 5 messages from the chat history
            messages += self.chat_history[-5:]

            # Call the OpenAI API with the prepared messages
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,  # Include the initial context and the last 5 chat messages
                max_tokens=300,
                temperature=0.5,
                stop=None
            )

            # Extract the response from GPT
            response_text = response.choices[0].message.content.strip()

            # Add the assistant's response to the chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text

        except openai.error.APIError as api_error:
            messagebox.showerror("Error", f"OpenAI API error: {str(api_error)}")
            return "Sorry, I'm currently unable to generate a response."

        except openai.error.RateLimitError as rate_error:
            messagebox.showerror("Error", f"OpenAI API rate limit reached: {str(rate_error)}")
            return "Sorry, the request rate has been exceeded. Please try again later."

        except Exception as e:
            messagebox.showerror("Error", f"Error in GPT response: Failed to connect. Probable cause: {str(e)}")
            return "Sorry, I couldn't generate a response at this time."

    def process_gpt_response(self, query):
        try:
            # Get the response from GPT-3
            response_text = self.ask_gpt(query)

            # Display the assistant's response in the transcript display
            self.transcript_display.insert(tk.END, f"Assistant: {response_text}\n", "assistant")
            self.transcript_display.see(tk.END)

            # Speak the response
            self.speak_text(response_text)

        except Exception as e:
            messagebox.showerror("Error", f"Error in GPT response: {str(e)}")

        finally:
            # Reset the waiting flag
            self.is_waiting_for_response = False

    def speak_text(self, text):
        self.is_speaking = True
        self.speaking_indicator.config(foreground="red")
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3_file:
            tts.save(temp_mp3_file.name)
            temp_mp3_file_path = temp_mp3_file.name

        if os.name == 'posix':  # macOS/Linux
            os.system(f"afplay {temp_mp3_file_path}")
        elif os.name == 'nt':  # Windows
            os.system(f"start {temp_mp3_file_path}")

        self.is_speaking = False
        self.speaking_indicator.config(foreground="grey")


# Run the application
if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Using a modern theme
    app = SpeechApp(root)
    root.mainloop()
