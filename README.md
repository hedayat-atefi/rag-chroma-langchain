# LangChain Example

This repository demonstrates an example use of the LangChain library to load documents from the web, split texts, create a vector store, and perform retrieval-augmented generation (RAG) utilizing a large language model (LLM). The example encapsulates a streamlined approach for splitting web-based documents, embedding the splits via OpenAI embeddings, saving those embeddings in a vector store, and then using those embeddings for context-dependent question-answering with a chat model.

## Features
- Document loading from a URL
- Recursive character-based document splitting
- Creating a vector store with Chroma
- Embedding documents using OpenAI Embeddings
- Setup of a retrieval-augmented generation (RAG) prompt
- Integration of a large language model (ChatOpenAI) for answering questions based on context derived from the vector store

## Requirements
- Python 3.x
- [langchain](https://github.com/langchain/langchain) library
- [langchain_core](https://github.com/langchain/langchain_core) library

Ensure you have the latest versions of these libraries installed to avoid compatibility issues.

## Installation

To install the necessary libraries for this example, you can use pip:

```bash
pip install langchain langchain_core
```

## Usage

The example is structured into several parts: loading documents, splitting texts, creating a vector store, embedding documents, and setting up a RAG chain for context-dependent question answering. Here is a brief outline on how to use each part:

### Document Loading
The example starts by loading a document from a specified URL using `WebBaseLoader`. The loaded document is then split into chunks using `RecursiveCharacterTextSplitter`, which helps in handling large documents by splitting them into manageable sizes.

### Creating a Vector Store
After splitting, the document chunks are embedded using `OpenAIEmbeddings`, and these embeddings are stored in a `Chroma` vector store. This allows for efficient retrieval of document sections relevant to a particular query.

### Embedding Single Document
For testing purposes, embedding a single document and adding it to the vector store is also demonstrated.

### Retrieval-Augmented Generation (RAG) Chain
The key feature of this example is setting up a RAG chain that utilizes the previously created vector store to retrieve context relevant to a given question. The chain integrates a prompt template, the ChatOpenAI model for language understanding, and an output parser, facilitating a context-aware question-answering system.

### Running the RAG Chain
Here's a simple example on how to run the chain:

```python
# Assuming the chain is already setup as demonstrated above
question = Question(__root__="What is LangChain?")
response = chain.run(question)
print(response)
```

Ensure that `Question` is properly formatted as expected by the RAG chain.

## Contribution

Contributions to this example are welcome. Please ensure to follow the standard contribution guidelines, including writing tests for new features and documenting your code appropriately.

## License

This example is open-sourced under the MIT License. See the LICENSE file for more details.

**Disclaimer**: This README provides an overview for educational purposes and a starting point for using LangChain and related libraries. The example code and setup instructions are subject to change based on updates to the dependencies and their APIs.
