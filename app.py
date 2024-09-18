import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma as ChromaStore
from dotenv import load_dotenv
from langchain.llms import OpenAI
import streamlit as st

def run_query(query, chat_history):
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    persist_directory = 'db'
    db = ChromaStore(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k": 4})
    search_results = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in search_results])
    full_query = f"{chat_history}\n\nContext: {context}\n\nUser Query: {query}"
    llm = OpenAI(api_key=openai_key, max_tokens=1500, temperature=0.2)
    llm_response = llm(full_query)
    return llm_response

role = """
You are Michael Scott, the eccentric and sometimes clueless boss from 'The Office'. 
Your responses should be humorous, overly confident, and reflect your unique management style, 
which includes awkward jokes, misunderstandings, and a deep need to be loved by everyone. 
Please use the context above to inform your response, but always keep it light-hearted, 
a bit clueless, and in the style of Michael Scott.
"""

st.title("Michael Scott Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        query = role + "\n\n" + user_input
        result = run_query(query, "\n".join(st.session_state.chat_history))
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Mr. Scott: {result}")
        
        for message in st.session_state.chat_history:
            st.write(message)