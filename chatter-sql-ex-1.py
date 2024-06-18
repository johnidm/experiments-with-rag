import os
import warnings
from operator import itemgetter

from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SAWarning
from langchain_community.agent_toolkits import create_sql_agent



warnings.filterwarnings(
    "ignore", r".*support Decimal objects natively, and SQLAlchemy", SAWarning
)
os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"


db = SQLDatabase.from_uri("sqlite:///db/chinook.db")

# print("Dialect:", db.dialect)
# print("Tables:", db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(
    llm,
    db,
)
response = chain.invoke(
    {"question": "How many employees are there"},
)

print(response)
print(db.run(response))
# chain.get_prompts()[0].pretty_print()


execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
response = chain.invoke({"question": "How many employees are there"})
print(response)


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

print(chain.invoke({"question": "How many employees are there"}))
print(chain.invoke({"question": "How many U2 albums are there?"}))
print(chain.invoke({"question": "How many Capital Inicial albums are there?"}))


agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
response = agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)
print(response)

response = agent_executor.invoke({"input": "Describe the playlisttrack table"})
print(response)
