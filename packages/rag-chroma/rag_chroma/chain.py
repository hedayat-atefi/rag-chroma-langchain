from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from typing import List, Any

class Question(BaseModel):
    """
    Typed input for questions. Used for chaining and validation.
    """
    __root__: str

def build_vectorstore_from_texts(
    texts: List[str],
    collection_name: str = "rag-chroma",
    embedding_model: Any = None,
) -> Chroma:
    """
    Initializes a Chroma vectorstore from in-memory texts.
    """
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()
    return Chroma.from_texts(
        texts,
        collection_name=collection_name,
        embedding=embedding_model,
    )

def build_retriever(vectorstore: Chroma):
    """
    Gets a retriever from a vectorstore.
    """
    return vectorstore.as_retriever()

def build_rag_chain(retriever):
    """
    Constructs a RAG chain: retrieval-augmented generation pipeline.
    """
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.with_types(input_type=Question)

def example():
    """
    Example: build a simple RAG chain for a sample text.
    """
    texts = ["harrison worked at kensho"]
    vectorstore = build_vectorstore_from_texts(texts)
    retriever = build_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    return rag_chain

# Uncomment the following to run a sample question:
# if __name__ == '__main__':
#     chain = example()
#     print(chain.invoke(Question(__root__='Where did Harrison work?')))

# --- DOCUMENTED WORKFLOWS FOR DOCUMENT LOADING AND SPLITTING ---

#
# To load, split and initialize a vectorstore from a URL, use the following snippets:
#
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()
#
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)
#
# vectorstore = Chroma.from_documents(
#     documents=all_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
# )
# retriever = vectorstore.as_retriever()
#
