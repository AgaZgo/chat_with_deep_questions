# Chat with Deep Questions podcast
Streamlit demo app that takes user's queries and generates answers based on the most relevant parts of podcast transcripts and chat history.

## Get OpenAI API key
- go to  https://platform.openai.com/account/api-keys
- click `+ Create new secret key` button
- enter a key name (optional) and confirm by clicking `Create secret key`


## Features:

1. Option to choose OpenAI GPT model from: 'gpt-3.5-turbo', 'gpt-4'. (Keep in mind that the latter generates much higher costs.)
2. Option to chat with single episode or with all episodes (for which transcripts are provided).
3. History of conversation is kept in chat memory.

## Tech stack:
- 🦜🔗 LangChain
- ![](st_icon.png) Streamlit