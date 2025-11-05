# rag-chroma-langchain

[![PyPI](https://img.shields.io/pypi/v/rag-chroma.svg?label=PyPI)](https://pypi.org/project/rag-chroma) [![Python](https://img.shields.io/pypi/pyversions/rag-chroma.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/github/license/hedayat-atefi/rag-chroma-langchain.svg)]

A developer-focused example and template that demonstrates Retrieval-Augmented Generation (RAG) using LangChain, Chroma vector store, and embeddings. The repository includes a small package template (packages/rag-chroma) and top-level examples showing how to ingest documents, build a Chroma collection, and wire a RAG pipeline.

Note: This README is provider-agnostic — it does not assume a single LLM provider. See package docs for provider-specific configuration.

Features
- Document loading (URL / file)
- Document chunking and text splitting
- Vectorstore creation with Chroma
- Embedding documents (provider-agnostic — configure your embedding provider)
- RAG chain scaffolding for context-aware QA
- A reusable LangChain template package for LangServe / LangChain projects (packages/rag-chroma)

Repository layout
- README.md — this file
- packages/rag-chroma/ — reusable template package with chain, ingest utilities, and a small FastAPI-based example
- scripts/ — ingestion and utility scripts
- notebooks/ — example notebooks (if present) for exploration and prototyping
- LICENSE — MIT license

Requirements
- Python 3.8+
- pip
- Recommended: create and use a virtual environment

Core dependencies (examples used in the repo)
- langchain
- langchain_core
- chromadb / chroma client (for Chroma vectorstore)
- an embeddings provider client (OpenAI, Cohere, etc.) — provider-agnostic approach

Installation
1. Clone the repo:
   git clone https://github.com/hedayat-atefi/rag-chroma-langchain.git
   cd rag-chroma-langchain

2. Create a virtual environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

3. Install required packages (adjust extras as needed):
   pip install -U pip
   pip install -e ".[all]"   # if the repo defines extras for pdf/html parsing, otherwise install core deps: langchain langchain_core chromadb

Configuration / environment variables
- Embeddings / LLM provider keys:
  - Configure your chosen embeddings and LLM provider credentials as environment variables as required by that provider (e.g., OPENAI_API_KEY for OpenAI).
- Chroma config:
  - By default, Chroma can create a local collection. If you run a remote Chroma service, set the corresponding connection variables used by the codepath you choose.
- Optional tracing:
  - If using LangChain tracing/monitoring (LangSmith), set LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT as needed.

Quick pointers for developers
- packages/rag-chroma/README.md contains package-specific instructions, including how to integrate the template into a LangChain project and how to run the included LangServe routes.
- scripts/ contains small CLI utilities to ingest files and URLs into a Chroma collection (supports .txt, .md/.mdx, .pdf, and HTTP(S) URLs).
- Use the ingestion utilities to upsert documents into a named Chroma collection, then initialize a RAG chain against that collection.

Testing & development
- Use your standard tooling (pytest, flake8/ruff, mypy) if tests or linters are added to the repo.
- Add tests for new features and maintainers should run them before merging changes.

Contributing
- Contributions are welcome. Please follow these guidelines:
  - Open issues for bugs or feature requests.
  - Create focused pull requests with descriptive titles and clear change descriptions.
  - Include tests and documentation for new features where applicable.
  - Follow existing code style and add type hints where helpful.

License
This project is licensed under the MIT License — see the LICENSE file for details.

Further reading and references
- packages/rag-chroma/README.md — package-specific usage and LangServe integration
- LangChain docs — https://langchain.com
- Chroma docs — https://www.trychroma.com
