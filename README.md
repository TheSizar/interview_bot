# Speech-Enabled Chat with GPT

This project implements a speech-enabled chatbot using OpenAI's GPT model and Google's Speech-to-Text API. It provides a graphical user interface for real-time speech recognition, text-to-speech output, and interaction with GPT.

## Project Structure

```
interview_bot/
│
├── myenv/                 # Virtual environment directory
├── scripts/
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_audio_format.py
│   │   └── test_google_api.py
│   └── speech_chat_gpt_question.py
├── .gitignore
├── README.md
├── requirements.txt
└── secrets.toml           # Configuration file for API keys
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/interview_bot
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. In the root directory of your project, you should find a file named `secrets.toml`. If it doesn't exist, create it.

2. Add your API keys and credentials to the `secrets.toml` file in the following format:

   ```toml
   OPEN_AI_API = "your_openai_api_key"

   [gcp_service_account]
   type = "service_account"
   project_id = "your_project_id"
   private_key_id = "your_private_key_id"
   private_key = "-----BEGIN PRIVATE KEY-----\nYour_Private_Key_Here\n-----END PRIVATE KEY-----\n"
   client_email = "your_client_email"
   client_id = "your_client_id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "your_client_x509_cert_url"
   universe_domain = "googleapis.com"
   ```

   Replace the placeholder values with your actual API keys and credentials.

## Usage

1. Ensure you're in the project root directory and your virtual environment is activated.

2. Run the main application:
   ```
   python scripts/speech_chat_gpt_question.py
   ```

3. The GUI will open. You can:
   - Test the Google and OpenAI APIs using the "Test Google & OpenAI APIs" button.
   - Select your input device from the dropdown menu.
   - Enter job context and general context in the provided text areas.
   - Upload relevant files using the "Upload File" button.
   - Start recording your speech using the "Start Recording" button.
   - Stop recording using the "Stop Recording" button.
   - View the conversation transcript in the main text area.

## Testing

You can run individual test scripts to verify API connections. Make sure you're in the project root directory:

1. Test OpenAI API:
   ```
   python scripts/tests/test_api.py
   ```

2. Test Google Speech-to-Text API:
   ```
   python scripts/tests/test_google_api.py
   ```

3. Test audio format and streaming:
   ```
   python scripts/tests/test_audio_format.py
   ```

## Troubleshooting

- If you encounter any issues with API authentication, double-check your `secrets.toml` file in the project root directory to ensure all credentials are correct.
- Make sure you have the necessary permissions for the Google Cloud project associated with your service account.
- If you experience audio-related issues, try selecting a different input device from the dropdown menu in the GUI.
- Ensure your virtual environment is activated when running the scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the GPT model
- Google Cloud for the Speech-to-Text API
- The developers of the various Python libraries used in this project

For any additional questions or issues, please open an issue in the GitHub repository.