import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI


import os

os.environ[
    "OPENAI_API_KEY"
] = "sk-proj-"

df = pd.read_csv("./csv/data.csv")
print(df.head())

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)


response = agent.run("Who are the top 10 artists with highest danceable songs?")
print(response)

print(agent.agent.llm_chain.prompt.template)

# agent = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#     df,
#     verbose=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
# )

# agent.invoke("how many rows are there?")
