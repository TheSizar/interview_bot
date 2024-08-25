import os
from openai import OpenAI
import streamlit as st

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
client_openai = OpenAI()


def test_chat_gpt():
    try:
        # Create a simple context and message history
        context = "You are a helpful assistant."
        chat_history = [{"role": "system", "content": context}]
        user_message = "Hello, how are you?"

        # Append user's message to chat history
        chat_history.append({"role": "user", "content": user_message})

        # Send a message to GPT
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            max_tokens=50,  # Limiting the response for testing
            temperature=0.3
        )

        # Get the response content
        gpt_response = response.choices[0].message.content.strip()
        print(f"ChatGPT Response: {gpt_response}")

        # Check if a response was received
        if gpt_response:
            print("ChatGPT function test passed!")
        else:
            print("ChatGPT function test failed: No response received.")
    except Exception as e:
        print(f"ChatGPT function test failed: {e}")


# Run the test
test_chat_gpt()
