import json
import toml
import tempfile
import os
from google.cloud import speech

# Load secrets from .streamlit/secrets.toml
with open('../../secrets.toml', 'r') as f:
    secrets = toml.load(f)

# Create a temporary file with the Google Cloud credentials
google_cloud_credentials = secrets["gcp_service_account"]

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
    json.dump(google_cloud_credentials, temp_file)
    temp_file_path = temp_file.name

# Set the environment variable to the path of the temporary file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

# Print the environment variable to verify it is correctly set
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")


def test_google_speech_api_with_real_audio():
    try:
        client_speech = speech.SpeechClient()

        # Replace this path with the path to your actual audio file
        audio_file_path = "/Users/lastowskicezary/Downloads/conference.wav"

        # Load the audio file into memory
        with open(audio_file_path, 'rb') as audio_file:
            audio_content = audio_file.read()

        # Create RecognitionAudio and RecognitionConfig
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,  # Match this with the actual sample rate of the audio file
            language_code="en-US"
        )

        # Perform the recognition request
        response = client_speech.recognize(config=config, audio=audio)

        # Check the response
        if response.results:
            print("Google Speech API test passed!")
            for result in response.results:
                print(f"Transcript: {result.alternatives[0].transcript}")
        else:
            print("Google Speech API test failed: No results.")
    except Exception as e:
        print(f"Google Speech API test failed: {e}")

# Run the test with an actual audio file
test_google_speech_api_with_real_audio()
