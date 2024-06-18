import os

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"


db = SQLDatabase.from_uri("sqlite:///db/fashion.db")
llm = ChatOpenAI(temperature=0)


db_agent = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

db_agent.invoke("how many rows are there?")
db_agent.invoke("how many entries of Adidas are present?")
db_agent.invoke(
    "how many XL products of Nike are there that have a rating of more than 4?"
)
db_agent.invoke(
    "Give all the details of Adidas which have a size of L and have a rating of more than 4.2"
)
