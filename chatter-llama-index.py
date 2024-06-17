import os


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

os.environ["OPENAI_API_KEY"] = "sk-proj-"

PERSIST_DIR = "./storage"
DOCS_DIR = "./pdfs"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


"""
If you intend to use for quering propouses use the following code:
```
query_engine = index.as_query_engine() 
```
"""

chat_engine = index.as_chat_engine()  # chat engine with memory


def main():
    while True:
        query = input("Enter a query: ")

        if query == "":
            break

        response = chat_engine.query(query)
        print(response)


if __name__ == "__main__":
    main()
