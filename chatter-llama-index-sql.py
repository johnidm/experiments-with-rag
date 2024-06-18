from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex


from sqlalchemy import text
import os

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"

engine = create_engine("sqlite:///db/movie.db")

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
sql_database = SQLDatabase(engine)


with engine.connect() as connection:
    results = connection.execute(text("select title from IMDB limit 5")).fetchall()
    print(results)

from llama_index.core.query_engine import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(sql_database=sql_database, llm=llm)
query_str = "Which title has the highest rating?"
response = query_engine.query(query_str)
print(response)


table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="IMDB")),
    (SQLTableSchema(table_name="earning")),
    (SQLTableSchema(table_name="genre")),
]

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

response = query_engine.query("Which title has the highest budget?")
print(response)
