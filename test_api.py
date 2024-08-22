import os
from openai import OpenAI
import streamlit as st

os.environ['OPENAI_API_KEY'] = st.secrets["OPEN_AI_API"]
client = OpenAI()
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Who is Putin"}]
    )
chat_response = response.choices[0].message.content
print(chat_response)