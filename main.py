from langchain.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI
import streamlit as st
import os
import time
import numpy as np
from dotenv import load_dotenv, find_dotenv

#find .env files
dotenv_path = find_dotenv()
#load environment variables
load_dotenv(dotenv_path)

st.markdown(
    r"""
    <style>
    .stDeployButton {
        visibility: hidden;
    }
    .stMainMenu {
        visibility: hidden;
    }
    .stStatusWidget {
        visibility: hidden;
    }
    .stAppToolbar {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

class SQLAgent:

    def __init__(self):
        #init env variables

        self.__db = SQLDatabase.from_uri(os.getenv("DB_URI"))
        self.__OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL")
        self.__AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
        self.__OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
        self.__AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

        #instantiate model and sql agent
        
        self.__db.get_usable_table_names()

        self.__llm = AzureChatOpenAI(
            azure_endpoint = self.__AZURE_ENDPOINT,
            api_key = self.__AZURE_OPENAI_API_KEY,
            model = self.__OPENAI_GPT_MODEL,
            api_version = self.__OPENAI_API_VERSION,
            verbose = "True"
        )

        self.__agent_executor = create_sql_agent(
            llm=self.__llm,
            toolkit=SQLDatabaseToolkit(db = self.__db, llm = self.__llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def get_agent(self):
        return self.__agent_executor
    
    def stream_data(self, text):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.1)


sql_agent = SQLAgent()
agent = sql_agent.get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def handle_user_prompts(prompt):
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    agent_response = agent.run(prompt)

    with st.chat_message("assistant"):
        st.write_stream(sql_agent.stream_data(agent_response))
    
    st.session_state.messages.append({"role": "assistant", "content": agent_response})



prompt = st.chat_input(placeholder = "Ask Something")
if prompt:
   handle_user_prompts(prompt)