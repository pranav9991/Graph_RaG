'''import os
import google.generativeai as genai
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load environment variables
load_dotenv()

# Get the Gemini API key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("⚠️ GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=api_key)

# Gemini LLM
LLM_MODEL = "gemini-2.0-flash-lite"
llm = genai.GenerativeModel(LLM_MODEL)

# Set default embedding model for LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
'''
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Configure local Ollama LLM (Mistral)
llm = Ollama(model="llama3.2:3b")

# Set the default embedding model for LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
