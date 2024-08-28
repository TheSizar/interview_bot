# Speech-Enabled Chat with GPT and Groq

This project implements a speech-enabled chatbot using OpenAI's GPT model or Groq's LLM, and Google's Speech-to-Text API. It provides a graphical user interface for real-time speech recognition, text-to-speech output, and interaction with AI models.

## Project Structure

```
interview_bot/
│
├── myenv/                 # Virtual environment directory
├── scripts/
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_audio_format.py
│   │   ├── test_google_api.py
│   │   └── test_ssl.py
│   ├── speech_chat_gpt_question.py
│   └── speech_chat_groq.py
├── .gitignore
├── README.md
├── requirements.txt
└── secrets.toml           # Configuration file for API keys
```

## File Descriptions

1. `speech_chat_gpt_question.py`: Main application script for GPT-based chat.
2. `speech_chat_groq.py`: Main application script for Groq-based chat.
3. `test_api.py`: Tests the OpenAI API connection.
4. `test_audio_format.py`: Tests audio streaming and format compatibility.
5. `test_google_api.py`: Tests the Google Speech-to-Text API connection.
6. `test_ssl.py`: Tests SSL connection to Groq API.
7. `secrets.toml`: Stores API keys and credentials (not tracked by git).
8. `requirements.txt`: Lists all Python dependencies.

## Prerequisites

- macOS (This application has been tested only on macOS)
- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Install Python:
   If you don't have Python installed, download and install it from [python.org](https://www.python.org/downloads/).

2. Open Terminal and clone this repository or download the source code:
   ```
   git clone <repository-url>
   ```

3. Navigate to the project directory:
   ```
   cd path/to/interview_bot
   ```

4. Create and activate a virtual environment:
   ```
   python3 -m venv myenv
   source myenv/bin/activate
   ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   This will install the following dependencies:
   - sounddevice==0.5.0
   - numpy==2.1.0
   - streamlit==1.37.1
   - toml==0.10.2
   - gTTS==2.5.3
   - google-cloud-speech==2.27.0
   - openai==1.42.0
   - python-docx==1.1.2
   - PyPDF2==3.0.1
   - ttkthemes==3.2.2
   - requests==2.31.0
   - urllib3==2.0.7
   - chardet==5.2.0
   - groq==0.4.2

   Note: Ensure you have the latest versions of these packages for optimal performance and compatibility.

## Configuration

1. In the root directory of your project, create a file named `secrets.toml`.

2. Add your API keys and credentials to the `secrets.toml` file in the following format:

   ```toml
   OPEN_AI_API = "your_openai_api_key"
   GROQ_API_KEY = "your_groq_api_key"

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

2. Run the main application (choose one):
   - For GPT-based chat:
     ```
     python scripts/speech_chat_gpt_question.py
     ```
   - For Groq-based chat:
     ```
     python scripts/speech_chat_groq.py
     ```

3. The GUI will open. You can:
   - Test the APIs using the "Test APIs" button.
   - Select your input device from the dropdown menu.
   - Enter job context and general context in the provided text areas.
   - Upload relevant files using the "Upload File" button.
   - Start recording your speech using the "Start Recording" button.
   - Stop recording using the "Stop Recording" button.
   - View the conversation transcript in the main text area.
   - Toggle dark mode using the "Toggle Dark Mode" button.

## GPT vs Groq

The project includes two main scripts: `speech_chat_gpt_question.py` and `speech_chat_groq.py`. The core functionality is similar, but they differ in the AI model they use:

1. GPT Version (`speech_chat_gpt_question.py`):
   - Uses OpenAI's GPT model.
   - Implements retry logic and error handling specific to OpenAI API.
   - References for implementation:
```python:scripts/speech_chat_gpt_question.py
startLine: 589
endLine: 630
```

2. Groq Version (`speech_chat_groq.py`):
   - Uses Groq's LLM.
   - Implements custom API calling method with SSL verification disabled.
   - Uses a different model: "mixtral-8x7b-32768".
   - References for implementation:
```scripts/speech_chat_groq
startLine: 542
endLine: 557
```

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

4. Test SSL connection to Groq API:
   ```
   python scripts/tests/test_ssl.py
   ```

## Troubleshooting

- If you encounter any issues with API authentication, double-check your `secrets.toml` file to ensure all credentials are correct.
- Make sure you have the necessary permissions for the Google Cloud project associated with your service account.
- If you experience audio-related issues, try selecting a different input device from the dropdown menu in the GUI.
- Ensure your virtual environment is activated when running the scripts.
- If you encounter SSL verification issues with the Groq API, the application disables SSL verification. Use caution when deploying in a production environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the GPT model
- Groq for their LLM API
- Google Cloud for the Speech-to-Text API
- The developers of the various Python libraries used in this project

For any additional questions or issues, please open an issue in the GitHub repository.
