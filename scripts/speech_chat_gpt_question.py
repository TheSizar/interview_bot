import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sounddevice as sd
import numpy as np
import queue
import tempfile
import json
import re
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
from datetime import datetime
import time
from requests.exceptions import RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import queue
import chardet

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to secrets.toml
secrets_path = os.path.join(script_dir, '..', 'secrets.toml')

# Load secrets from the secrets.toml
try:
    with open(secrets_path, 'r') as f:
        secrets = toml.load(f)
except FileNotFoundError:
    print(f"Error: secrets.toml not found at {secrets_path}")
    print("Please ensure the secrets.toml file exists in the correct location.")
    exit(1)

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

# Suppress ALTS warning
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'

class AudioLevelMeter(tk.Canvas):
    def __init__(self, master, device_index, **kwargs):
        super().__init__(master, **kwargs)
        self.device_index = device_index
        self.levels = [0] * 20
        self.configure(bg="#2E2E2E")  # Set a dark background for the meter
        self.create_rectangles()
        self.update_meter()

    def create_rectangles(self):
        self.bars = []
        for i in range(20):
            bar = self.create_rectangle(i * 15 + 5, 20, i * 15 + 15, 80, fill="gray", outline="")
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
        self.dark_mode = False
        self.volume = 1.0

        # Initialize style
        self.style = ttk.Style(self.root)
        self.style.theme_use('arc')

        self.context = ""
        self.chat_history = [{"role": "system", "content": "Context not set."}]
        self.stop_signal = False
        self.uploaded_files = []
        self.lock = threading.Lock()
        self.resume_content = ""

        # Initialize device list and names
        self.device_list = sd.query_devices()
        self.device_names = [device['name'] for device in self.device_list]

        self.setup_gui()
        self.apply_theme()

    def setup_gui(self):
        self.style = ttk.Style()
        self.style.theme_use('arc')

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top frame (API Test, Mic Check, Dark Mode Toggle)
        top_frame = ttk.Frame(main_frame, padding="5")
        top_frame.pack(fill=tk.X)

        self.device_var = tk.StringVar(self.root)
        if self.device_names:  # Check if device_names is not empty
            self.device_var.set(self.device_names[0])
        else:
            print("Warning: No audio devices found.")
            self.device_var.set("No devices available")

        self.device_menu = ttk.Combobox(top_frame, textvariable=self.device_var,
                                        values=self.device_names, state="readonly", width=30)
        self.device_menu.pack(side=tk.LEFT, padx=5)

        self.test_api_button = ttk.Button(top_frame, text="Test APIs", command=self.test_apis)
        self.test_api_button.pack(side=tk.LEFT, padx=5)

        self.dark_mode_button = ttk.Button(top_frame, text="Toggle Dark Mode", command=self.toggle_dark_mode)
        self.dark_mode_button.pack(side=tk.RIGHT, padx=5)

        # Audio level meter
        self.audio_level_meter = AudioLevelMeter(top_frame, device_index=self.device_names.index(self.device_var.get()),
                                                 width=310, height=100)
        self.audio_level_meter.pack(side=tk.LEFT, padx=5)

        # Middle frame (Context & File Upload)
        middle_frame = ttk.Frame(main_frame, padding="5")
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

        # File Upload
        file_frame = ttk.Frame(middle_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        self.upload_button = ttk.Button(file_frame, text="Upload File", command=self.upload_file, width=15)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.file_list = ttk.Treeview(file_frame, columns=("File",), show="headings", height=2)
        self.file_list.heading("File", text="Uploaded Files")
        self.file_list.column("File", width=200)
        self.file_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Bottom frame (Chat)
        bottom_frame = ttk.Frame(main_frame, padding="5")
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(button_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.speaking_indicator = ttk.Label(button_frame, text="●", foreground="grey")
        self.speaking_indicator.pack(side=tk.LEFT, padx=5)

        # Transcript Display (iMessage Style)
        self.transcript_display = scrolledtext.ScrolledText(bottom_frame, height=20, width=80, wrap=tk.WORD,
                                                            font=('Arial', 12))
        self.transcript_display.pack(fill=tk.BOTH, expand=True)
        self.transcript_display.tag_configure("user", foreground="white", background="#007AFF", justify="right",
                                              font=('Arial', 12, 'bold'), lmargin1=10, rmargin=10)
        self.transcript_display.tag_configure("assistant", foreground="black", background="#E5E5EA", justify="left",
                                              font=('Arial', 12), lmargin1=10, rmargin=10)
        self.transcript_display.tag_configure("timestamp", foreground="gray", font=('Arial', 10, 'italic'))

        # Loading indicator
        self.loading_label = ttk.Label(bottom_frame, text="", font=('Arial', 12, 'italic'))
        self.loading_label.pack()

        self.root.after_idle(self.root.update_idletasks)

        # Set up custom styles for dark mode compatibility
        self.style.configure("Custom.Treeview", rowheight=25)
        self.style.configure("Custom.TCombobox", selectbackground='#0078D7', selectforeground='white')
        self.style.configure("Custom.TButton", padding=5)
        self.style.configure("Custom.TEntry", fieldbackground="white", foreground="black")
        self.style.configure("Custom.TCombobox", fieldbackground="white", foreground="black",
                             selectbackground='#0078D7', selectforeground='white')
        self.style.configure("Custom.Treeview", fieldbackground="white", background="white", foreground="black")
        self.style.map('Custom.Treeview', background=[('selected', '#0078D7')], foreground=[('selected', 'white')])

    def apply_theme(self):
        bg_color = "#2E2E2E" if self.dark_mode else "white"
        fg_color = "white" if self.dark_mode else "black"
        button_fg_color = "black"  # Set button text color to black in dark mode
        entry_bg = "#3E3E3E" if self.dark_mode else "white"
        entry_fg = "white" if self.dark_mode else "black"
        entry_border_color = "white"  # Add a white border for the text boxes

        self.root.configure(bg=bg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("Custom.TButton", background=bg_color, foreground=button_fg_color)
        self.style.configure("Custom.TEntry", fieldbackground=entry_bg, foreground=entry_fg,
                             bordercolor=entry_border_color)
        self.style.configure("Custom.TCombobox", fieldbackground=entry_bg, foreground=entry_fg,
                             selectbackground='#0078D7', selectforeground='white')
        self.style.configure("Custom.Treeview", fieldbackground=entry_bg, background=entry_bg, foreground=fg_color)
        self.style.map('Custom.Treeview', background=[('selected', '#0078D7')], foreground=[('selected', 'white')])

        self.transcript_display.configure(bg=entry_bg, fg=fg_color, insertbackground=fg_color, bd=2, relief="solid",
                                          highlightbackground=entry_border_color)
        self.job_input.configure(bg=entry_bg, fg=fg_color, insertbackground=fg_color, bd=2, relief="solid",
                                 highlightbackground=entry_border_color)
        self.context_input.configure(bg=entry_bg, fg=fg_color, insertbackground=fg_color, bd=2, relief="solid",
                                     highlightbackground=entry_border_color)

        self.transcript_display.tag_configure("user", foreground="white",
                                              background="#0056b3" if self.dark_mode else "#007AFF")
        self.transcript_display.tag_configure("assistant", foreground="black",
                                              background="#4a4a4a" if self.dark_mode else "#E5E5EA")
        self.transcript_display.tag_configure("timestamp", foreground="#b0b0b0" if self.dark_mode else "gray")

        for widget in [self.test_api_button, self.dark_mode_button, self.upload_button,
                       self.start_button, self.stop_button, self.clear_button]:
            widget.configure(style="Custom.TButton")

        self.device_menu.configure(style="Custom.TCombobox")
        self.file_list.configure(style="Custom.Treeview")


    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        bg_color = "#2E2E2E" if self.dark_mode else "white"
        fg_color = "white" if self.dark_mode else "black"

        # Configure root and style
        self.root.configure(bg=bg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TButton", background=bg_color, foreground=fg_color)

        # Configure text widgets
        self.transcript_display.configure(bg=bg_color, fg=fg_color)
        self.job_input.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)
        self.context_input.configure(bg=bg_color, fg=fg_color, insertbackground=fg_color)

        # Reconfigure chat message tags
        self.transcript_display.tag_configure("user", foreground="white",
                                              background="#0056b3" if self.dark_mode else "#007AFF")
        self.transcript_display.tag_configure("assistant", foreground="black",
                                              background="#4a4a4a" if self.dark_mode else "#E5E5EA")
        self.transcript_display.tag_configure("timestamp", foreground="#b0b0b0" if self.dark_mode else "gray")

        # Update other widgets
        self.file_list.configure(style="Custom.Treeview")
        self.style.configure("Custom.Treeview",
                             background=bg_color,
                             fieldbackground=bg_color,
                             foreground=fg_color)
        self.style.map('Custom.Treeview', background=[('selected', '#0078D7')])

        self.device_menu.configure(style="Custom.TCombobox")
        self.style.configure("Custom.TCombobox",
                             fieldbackground=bg_color,
                             background=bg_color,
                             foreground=fg_color)

        # Update loading label
        self.loading_label.configure(foreground=fg_color)


    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if file_path:
            self.loading_label.config(text="Uploading file...")
            self.root.update_idletasks()
            threading.Thread(target=self.process_file, args=(file_path,), daemon=True).start()

    def process_file(self, file_path):
        try:
            print(f"Starting to process file: {file_path}")
            self.resume_content = self.read_file_content(file_path)
            if not self.resume_content:
                raise ValueError("No content could be extracted from the file.")
            print(f"File content read successfully")
            file_name = os.path.basename(file_path)
            self.uploaded_files.append(file_name)
            print(f"File {file_name} added to uploaded_files list")
            self.root.after(0, self.update_file_list)
            self.root.after(0, lambda: self.loading_label.config(text=""))
            self.root.after(0, lambda: messagebox.showinfo("File Uploaded", f"File '{file_name}' has been uploaded successfully."))
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            error_message = str(e)
            self.root.after(0, lambda: self.loading_label.config(text=""))
            self.root.after(0, lambda: messagebox.showerror("Upload Error", f"Failed to upload file: {error_message}"))

    def update_file_list(self):
        self.file_list.delete(*self.file_list.get_children())
        for file in self.uploaded_files:
            self.file_list.insert("", "end", values=(file,))

    def read_file_content(self, file_path):
        try:
            # Detect the file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Read the file content using the detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return f"Unable to read file content. Error: {str(e)}"

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
                model="chatgpt-4o-latest",
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
        self.chat_history = []
        threading.Thread(target=self.listen_and_transcribe, daemon=True).start()


    def clear_chat(self):
        self.transcript_display.delete(1.0, tk.END)
        self.chat_history = [{"role": "system", "content": "Context not set."}]
    
    def is_behavioral_question(self, text):
        behavioral_patterns = [
            r"tell me about a time when",
            r"describe a situation where",
            r"give me an example of",
            r"how did you handle",
            r"what do you do when",
        ]
        return any(re.search(pattern, text.lower()) for pattern in behavioral_patterns)

    def prepare_initial_context(self):
        context = self.context_input.get("1.0", tk.END).strip()
        job_context = self.job_input.get("1.0", tk.END).strip()

        # Extract key information from resume
        key_projects = self.extract_key_projects(self.resume_content)
        key_skills = self.extract_key_skills(self.resume_content)

        system_message = (
            f"You are embodying the person described in the following resume. Answer all questions in the first person, "
            f"using 'I' statements. Always use bullet points in your responses. Here's your background:\n\n"
            f"Resume Content:\n{self.resume_content}\n\n"
            f"Key Projects:\n{key_projects}\n\n"
            f"Key Skills:\n{key_skills}\n\n"
            f"Job Description:\n{job_context}\n\n"
            f"General Context: {context}\n\n"
            f"When answering questions:\n"
            f"1. Always use bullet points for each main point or reason.\n"
            f"2. Draw specific examples from the key projects and skills listed.\n"
            f"3. Relate your answers to the job description whenever possible.\n"
            f"4. For behavioral questions, use the STAR method (Situation, Task, Action, Result) in bullet point format.\n"
            f"5. Be concise but detailed, aiming for 3-5 bullet points per answer.\n"
        )

        return {"role": "system", "content": system_message}

    def extract_key_projects(self, resume_content):
        # This is a simple extraction method. You might need to adjust based on your resume format
        projects = re.findall(r'Project:.*?(?=Project:|$)', resume_content, re.DOTALL)
        return "\n".join([f"• {project.strip()}" for project in projects])

    def extract_key_skills(self, resume_content):
        # This is a simple extraction method. You might need to adjust based on your resume format
        skills_section = re.search(r'Skills:.*', resume_content, re.DOTALL)
        if skills_section:
            skills = skills_section.group().split(':')[1].split(',')
            return "\n".join([f"• {skill.strip()}" for skill in skills])
        return ""

    def prepare_behavioral_prompt(self, query):
        prompt = f"""
        As the person described in the resume, answer this behavioral question: {query}

        Use the STAR method (Situation, Task, Action, Result) to structure your response. 
        Draw from one of your key projects or experiences listed in your resume.

        Your response should follow this structure, using bullet points:
        • Situation:
          - Briefly describe the context
        • Task:
          - Explain what you needed to do
        • Action:
          - Detail the steps you took (use sub-bullets if necessary)
        • Result:
          - Highlight the outcome
          - Mention what you learned or how it relates to the job you're applying for

        Ensure your answer is specific, drawing directly from your resume, and relates to the job description.
        Begin your response now, speaking as yourself based on your resume.
        """
        return prompt

    def prepare_general_prompt(self, query):
        prompt = f"""
        As the person described in the resume, answer this question: {query}

        Remember to:
        • Use bullet points for each main point or reason
        • Draw specific examples from your key projects and skills
        • Relate your answer to the job description when relevant
        • Be concise but detailed, aiming for 3-5 main bullet points

        Begin your response now, speaking as yourself based on your resume and the job description.
        """
        return prompt

    def stop_recording(self):
        self.stop_signal = True
        try:
            audio_queue.put(None)
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
        """Determine if the text is a question or a behavioral query."""
        question_words = ["who", "what", "where", "when", "why", "how", "is", "are", "do", "does", "can", "could",
                          "would", "should", "tell me about"]
        return text.strip().endswith("?") or text.lower().startswith(tuple(question_words))

    def ask_gpt(self, query):
        print(f"Asking GPT: {query}")
        max_retries = 3
        backoff_factor = 0.5

        for attempt in range(max_retries):
            try:
                if self.is_behavioral_question(query):
                    prompt = self.prepare_behavioral_prompt(query)
                else:
                    prompt = self.prepare_general_prompt(query)

                messages = [
                    self.initial_context,
                    {"role": "user", "content": prompt}
                ]

                response = client_openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content.strip()
                self.chat_history.append({"role": "assistant", "content": response_text})
                print(f"GPT response received, length: {len(response_text)}")
                return response_text

            except RequestException as e:
                print(f"RequestException in ask_gpt (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "I'm sorry, but I'm having trouble connecting right now. Please try again in a moment."
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                print(f"Unexpected error in ask_gpt: {e}")
                return "An unexpected error occurred. Please try again."

    def display_message(self, sender, message, tag):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcript_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.transcript_display.insert(tk.END, f"{sender}: {message}\n", tag)
        self.transcript_display.see(tk.END)

    def process_gpt_response(self, query):
        try:
            print(f"Processing GPT response for query: {query}")
            self.loading_label.config(text="Processing...")
            self.root.update_idletasks()
            response_text = self.ask_gpt(query)
            print(f"Received response from GPT: {response_text[:50]}...")  # Print first 50 chars of response
            self.display_message("Assistant", response_text, "assistant")
            self.speak_text(response_text)
        except Exception as e:
            print(f"Error in GPT response: {str(e)}")
            messagebox.showerror("Error", f"Error in GPT response: {str(e)}")
        finally:
            print("Resetting is_waiting_for_response flag")
            self.is_waiting_for_response = False
            self.loading_label.config(text="")
            self.root.update_idletasks()

    def speak_text(self, text):
        self.root.after(0, lambda: self.speaking_indicator.config(foreground="red"))
        self.is_speaking = True
        
        def speak_and_reset():
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3_file:
                tts.save(temp_mp3_file.name)
                temp_mp3_file_path = temp_mp3_file.name

            if os.name == 'posix':  # macOS/Linux
                os.system(f"afplay {temp_mp3_file_path}")
            elif os.name == 'nt':  # Windows
                os.system(f"start {temp_mp3_file_path}")

            self.is_speaking = False
            self.root.after(0, lambda: self.speaking_indicator.config(foreground="grey"))
            print("Finished speaking, ready for next question")
            self.is_waiting_for_response = False

        threading.Thread(target=speak_and_reset, daemon=True).start()

# Run the application
if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Using a modern theme
    app = SpeechApp(root)
    root.mainloop()
