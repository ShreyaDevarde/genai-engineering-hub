"""
Project: Understanding LLM Tokenization

Problem Statement:
    Large Language Models (LLMs) do not process raw text directly.
    Instead, they convert text into smaller units called tokens,
    which are then mapped to numerical token IDs before being
    passed to the model. Understanding tokenization is essential
    because model context limits, latency, and API costs are all
    measured in tokens rather than words.

Objective:
    Demonstrate how different types of text are tokenized using
    OpenAI's tiktoken library and inspect the resulting token IDs
    and token counts.

Concepts Covered:
    - Tokenization
    - Token IDs
    - LLM Input Processing
    - Context Window Management
    - Token-Based Pricing
    - Text Encoding

Workflow:
    1. Initialize a tokenizer using the 'cl100k_base' encoding.
    2. Provide sample inputs containing:
       - Common words
       - Sentences
       - Long uncommon words
       - Numeric values
       - Emojis
    3. Convert each text input into token IDs.
    4. Calculate the number of tokens generated.
    5. Display the original text, token count, and token IDs.

Expected Learning Outcome:
    - Understand how text is transformed into tokens before
      reaching an LLM.
    - Observe that token counts do not directly correspond to
      word counts.
    - Learn that different text types (words, numbers, emojis,
      special characters) produce different tokenization patterns.
    - Build foundational knowledge required for Prompt Engineering,
      RAG, LangChain, LlamaIndex, AI Agents, and cost optimization.

Why This Matters:
    Every interaction with an LLM begins with tokenization.
    Token counts directly impact:
        - API costs
        - Context window usage
        - Response latency
        - Model performance

    Understanding tokenization is a fundamental skill for
    LLM Engineers, GenAI Developers, and AI Solution Architects.
"""

import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # same encoding as GPT-4

texts = [
    "Hello",
    "Hello world",
    "LangChain is a framework for building LLM applications.",
    "supercalifragilisticexpialidocious",
    "1234567890",
    "🚀🤖🔥"
]

print(f"{'Text':<50} {'Tokens':>8} {'Token IDs'}")
print("-" * 80)
for text in texts:
    tokens = encoder.encode(text)
    print(f"{text:<50} {len(tokens):>8}   {tokens}")