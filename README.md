# SlowGroq

This module provides a wrapper for `ChatGroq` to control the rate of streaming token output.

## Installation

Ensure you have the `langchain-core` and related dependencies installed.

```bash
pip install langchain-core
```

## Usage

Import `SlowGroq` and wrap your `ChatGroq` instance:

```python
from langchain_groq import ChatGroq

# Initialize your Groq chat model
groq_model = ChatGroq(model="llama-3.1-8b-instant")

# Wrap with throttling (e.g., 5 tokens per second)
slow_groq = SlowGroq(groq_model, tokens_per_second=5)

# Stream messages slowly
for chunk in slow_groq.stream(messages):
    print(chunk.text, end="", flush=True)
```

## Parameters

- `groq_model`: An instance of `ChatGroq`.
- `tokens_per_second`: Optional float specifying the desired token rate.
