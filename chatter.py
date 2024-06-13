import os


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate


open_ai_key = "sk-"
os.environ["OPENAI_API_KEY"] = open_ai_key


SPLITTER = "character"
LLM = "gpt-4"
EMBEDDING = "llama3"


def load_documents():
    directory_path = "pdfs"

    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents


def get_text_chunks(text):
    splitters = {
        "character": RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100),
        "semantic": SemanticChunker(OpenAIEmbeddings()),
    }

    print(f"Selected splitter: {SPLITTER}")

    text_splitter = splitters[SPLITTER]

    chunks = text_splitter.split_documents(text)
    return chunks


def get_vector_store_retriever(text_chunks):
    embeddings = {
        "openai": OpenAIEmbeddings(),
        "llama3": OllamaEmbeddings(model="llama3"),
    }

    print(f"Selected embedding: {EMBEDDING}")

    vectorstore = Chroma.from_documents(
        documents=text_chunks, embedding=embeddings[EMBEDDING]
    )
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm():
    models = {
        "gpt-4": ChatOpenAI(model_name="gpt-4", temperature=0),
        "gpt-3.5-turbo": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        "llama3": Ollama(model="llama3"),
    }

    print(f"Selected model: {LLM}")

    llm = models[LLM]
    return llm


def get_chain(retriever):
    llm = get_llm()

    template = """Baseado no contexto abaixo:
    {context}

    Responda a seguinte pergunta:
    {question}

    Se você não sabe a resposta, apenas diga que você não sabe.
    """

    prompt_custom = PromptTemplate.from_template(template)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_custom
        | llm
        | StrOutputParser()
    )
    return chain


def build_chain():
    print("Loading data...")
    documents = load_documents()

    print("Splitting data...")
    chunks = get_text_chunks(documents)

    print("Creating vector store...")
    retriever = get_vector_store_retriever(chunks)

    print("Creating RAG chain...")
    chain = get_chain(retriever)

    return chain


def main():
    chain = build_chain()

    while True:
        query_text = input("Question (press Enter to exit): ")

        if query_text == "":
            break

        response = chain.invoke(query_text)
        print(response)


if __name__ == "__main__":
    main()
