"""
Generate a response from a Groq-hosted Large Language Model (LLM)
using the Chat Completions API and display token usage statistics.

Workflow:
    1. Load environment variables from a .env file.
    2. Read the GROQ_API_KEY securely from the environment.
    3. Create a Groq client instance for API communication.
    4. Send a chat completion request to the selected LLM.
    5. Provide system and user messages as conversational context.
    6. Receive the model-generated response.
    7. Print the generated answer.
    8. Display response metadata including model name and token usage.

Model:
    llama-3.1-8b-instant

Required Environment Variables:
    GROQ_API_KEY : Valid Groq API key.

Returns:
    None

Outputs:
    - Model-generated text response.
    - Model name used for inference.
    - Prompt token count.
    - Completion token count.
    - Total token count.

Purpose:
    Demonstrates the fundamental interaction between an application
    and a Large Language Model (LLM) without using higher-level
    frameworks such as LangChain, LlamaIndex, or Agno. This serves
    as the foundation for understanding RAG systems, AI agents,
    prompt engineering, and GenAI application development.
"""

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is a Large Language Model in 3 sentences?"}
    ]
)

print(response.choices[0].message.content)
print("\n--- Response Metadata ---")
print("Model used     :", response.model)
print("Prompt tokens  :", response.usage.prompt_tokens)
print("Response tokens:", response.usage.completion_tokens)
print("Total tokens   :", response.usage.total_tokens)