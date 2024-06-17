from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os


import warnings
from sqlalchemy.exc import SAWarning

warnings.filterwarnings(
    "ignore", r".*support Decimal objects natively, and SQLAlchemy", SAWarning
)

os.environ["OPENAI_API_KEY"] = "sk-proj-"


db_uri = "sqlite:///./db/chinook.db"

db = SQLDatabase.from_uri(db_uri)


template_sql = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""


prompt_sql = ChatPromptTemplate.from_template(template_sql)


def get_schema(_):
    schema = db.get_table_info()
    return schema


llm = ChatOpenAI()

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt_sql
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""


prompt_response = ChatPromptTemplate.from_template(template)


def run_query(query):
    return db.run(query)


full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm
)


if __name__ == "__main__":
    while True:
        user_question = input("Enter a question: ")

        if user_question == "":
            break

        result = full_chain.invoke({"question": user_question})
        print(result.content)
