# Experiments with RAG

This repository contains a collection of small projects designed to explore the application of RAG (Retrieval-Augmented Generation) in real-world scenarios. 

The primary objective is to acquire hands-on experience and develop a deeper understanding of how to effectively utilize RAG in tackling practical problems

#### Install the requirements using pip:

```
pip install -r requirements.txt
```

## Chatter (chatter.py)

A simple question and answer application using LangChain framework and Chroma as the vector database.

The documents need to be in PDF format.

#### How to use

- Put your PDF files in the `pdfs` folder.
- Execute the script running the command `python chatter.py`.
- Make questions about the document's context.

Run the command `python chatter.py` to start the application.

## ChatterQA (chatter-qa.py)

Using QA LangChain module. 

Works the same way as `chatter.py`, but using a different module.

Run the command `python chatter-qa.py` to start the application.

## ChatterSQL (chatter-sql.py, chatter-sql-ex.py, chatter-sql-ex-1.py, chatter-llama-index-sql.py)

This is a example of how to use RAG to answer questions in a SQL database.

There are tow database in the `db` folder:
- `chinook.db`: A SQLite database about music store.
- `movies.db`: A SQLite database about movies store.

Run the command `python chatter-sql.py` to start the application.

## Chatter LLama Index (chatter-llama-index.py)

A simples question and anwser application using LlamaIndex framework.

This application works the same way as `chatter.py`, but using a different framework keeping the chat history.

Run the command `python chatter-llama-index.py` to start the application.

## References

- [LangChain - Q&A with RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)
- [LangChain - Q&A over SQL + CSV](https://python.langchain.com/v0.1/docs/use_cases/sql/)
- [LlamaIndex - RAG Applications for Production](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
