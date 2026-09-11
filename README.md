# RAG Pipeline & API Experiments

A small personal repository for experimenting with **LLMs, embeddings, API calls, local models, and Retrieval-Augmented Generation (RAG)**.

This repository is mainly a learning playground rather than a production-ready project. The scripts are intentionally simple so I can experiment with different models, APIs, and RAG components independently.

## Main Focus

The main experiment in this repository is [`rag_pipeline.py`](./rag_pipeline.py).

It implements a simple end-to-end RAG pipeline:

```text
PDF documents
     ↓
Document loading
     ↓
Text splitting
     ↓
Gemini embeddings
     ↓
FAISS vector store
     ↓
Similarity retrieval
     ↓
Prompt with retrieved context
     ↓
Gemini LLM
     ↓
Answer
```

The goal is to understand the basic building blocks of a RAG system rather than hide everything behind a high-level framework.

## `rag_pipeline.py`

The pipeline currently:

1. Loads PDF files from the `./data` directory.
2. Splits documents into chunks using `RecursiveCharacterTextSplitter`.
3. Generates embeddings with Google's Gemini embedding model.
4. Stores the embeddings in a FAISS vector store.
5. Creates a retriever using similarity search with a score threshold.
6. Passes the retrieved context to a Gemini chat model.
7. Generates an answer using only the retrieved context.

The current configuration uses:

* Chunk size: `1200`
* Chunk overlap: `200`
* Top-k retrieval: `5`
* Similarity score threshold: `0.3`
* Embedding model: `gemini-embedding-2-preview`
* Chat model: `gemini-3.7-flash`
* Temperature: `0`

The prompt is intentionally strict: if the answer cannot be found in the retrieved context, the model is instructed to say that it does not know based on the provided context. This is meant to reduce guessing and keep the experiment focused on retrieval quality.

## Other Experiments

### `embed_api.py`

A small experiment with the Gemini API for generating embeddings.

It sends a simple text query to the embedding API and prints the resulting embedding vector.

### `ollama_chat.py`

A minimal experiment with a local model through Ollama.

The current script sends a prompt to `qwen3.5:2b` and streams the generated response token by token.

### `lmstudio.py`

An experiment with a local model served through LM Studio's OpenAI-compatible API.

It connects to:

```text
http://127.0.0.1:1234/v1
```

and sends a simple chat completion request using the OpenAI Python client.

## Project Structure

```text
Rag_pipeline_and_api_call/
│
├── rag_pipeline.py      # Main RAG experiment
├── embed_api.py         # Gemini embedding experiment
├── ollama_chat.py       # Local LLM experiment with Ollama
├── lmstudio.py          # Local LLM experiment with LM Studio
└── README.md
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages according to the scripts you want to experiment with.

For the RAG pipeline, the main dependencies include:

```bash
pip install \
  langchain-community \
  langchain-text-splitters \
  langchain-google-genai \
  langchain-openai \
  faiss-cpu \
  python-dotenv \
  unstructured
```

Depending on the PDF files and your environment, additional dependencies may be required by `unstructured`.

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do **not** commit your `.env` file or expose your API key publicly.

## Running the RAG Pipeline

Put your PDF files inside:

```text
data/
```

Then run:

```bash
python rag_pipeline.py
```

The script will load the PDFs, build the vector store, ask for a question, and print the generated answer.

Example:

```text
Question: What is ...
```

## Learning Goals

This repository is mainly for exploring questions such as:

* How does a basic RAG pipeline actually work?
* How should documents be chunked?
* How do embeddings affect retrieval quality?
* How does similarity search work?
* How can FAISS be used as a local vector store?
* How should retrieved context be passed to an LLM?
* How different local and cloud models behave?
* What is the difference between Ollama, LM Studio, and API-based models?

## Notes

This project is intentionally experimental and may change frequently.

The code is written primarily for learning, testing ideas, and understanding the underlying components of LLM applications.

More experiments will be added as I continue learning.
