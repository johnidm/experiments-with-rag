import os

from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["OPENAI_API_KEY"] = "sk-proj-"

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200)

pdf_link = "pdfs/ticket_to_ride_manual_table_games.pdf"

loader = PyPDFLoader(pdf_link, extract_images=False)
pages = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(pages)

db = Chroma.from_documents(
    chunks, embedding=embeddings_model, persist_directory="index"
)

vectordb = Chroma(persist_directory="index", embedding_function=embeddings_model)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chain = load_qa_chain(llm, chain_type="stuff")


def main():
    while True:
        question = input("Enter your question: ")

        if question == "":
            break

        context = retriever.invoke(question)

        answer = chain.invoke(
            {"input_documents": context, "question": question},
            return_only_outputs=True,
        )

        print(answer)


if __name__ == "__main__":
    main()
