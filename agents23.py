import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os


## initialize arxiv and wikipedia tools
# create an arxiv wrapper to define how many results to fetch and hwo many content from each 
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200) 
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# creating a duck duck go serach tool for general purpose search
search= DuckDuckGoSearchRun(name="Search")


# Streamlit UI Setup
st.title("Search engine (Gen AI app) using Tools and Agents")

# Creating sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Please enter Groq API key:" , type="password")

## session state for chat messages

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi Iam a chatbot who can search the web. How can i help you?"
        }
    ]
if prompt := st.chat_input(placeholder="Please write your question here..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key = api_key, model_name = "gemma2-9b-it")

    tools = [search,arxiv,wiki]

    # we will create an agent that uses .zero shot react description

    search_agent = initialize_agent(
        tools,
        llm,
        agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True
    )

    # get and display the response 
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)

        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        st.session_state.messages.append({
            'role': 'assistant',
            'content': 'response'
        })
        # display the asssistant response in the chat 
        st.write(response)