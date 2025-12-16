import os

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    base_url='https://ollama.com',
    model='gpt-oss:120b',
    lc_secrets={
        'headers' : {'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    }
)

# Created By Amit Mahapatra