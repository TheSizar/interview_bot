import queue

import numpy as np
import sounddevice as sd
from google.cloud import speech
import tempfile
import json
import os
import toml

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

def audio_callback(indata, frames, time, status):
    """This callback function will be called for each audio block."""
    q.put(indata.copy())

def stream_audio():
    global q
    q = queue.Queue()

    # Initialize Google Speech-to-Text client
    client = speech.SpeechClient()

    # Configuration for the Google Speech-to-Text API
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,  # If you want to see partial transcriptions as the user speaks
    )

    # Set a smaller blocksize for lower latency
    with sd.InputStream(samplerate=44100, channels=1, callback=audio_callback, blocksize=1024):
        print("Start speaking...")

        # Create a generator that yields audio chunks from the queue
        audio_generator = stream_generator()

        requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)

        # Perform streaming recognition
        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        # Process and print the responses
        try:
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        print("Transcript: {}".format(result.alternatives[0].transcript))
        except Exception as e:
            print(f"Error during streaming: {e}")

def stream_generator():
    """A generator that yields chunks of audio data."""
    while True:
        data = q.get()
        if data is None:
            break
        # Convert the audio chunk to int16 and then to bytes
        audio_data = (data * 32767).astype(np.int16)
        yield audio_data.tobytes()

# Start the streaming recognition
stream_audio()