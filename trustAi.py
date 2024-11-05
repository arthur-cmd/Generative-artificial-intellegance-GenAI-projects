from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st 


st.set_page_config(page_title="Arthur.ai", page_icon="🤖")
st.title("Arthur's Oracle")

def get_response(user_query,chat_history):
  
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt=ChatPromptTemplate.from_template(template)

    model= OllamaLLM(model="gemma2")

    chain = prompt | model | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello,I am a bot. How can I help you?"),
    ]



# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)



# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
       ai_response= st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(ai_response))


# template="""Question: {question}

# Answer: Let's think step by step."""

# prompt=ChatPromptTemplate.from_template(template)

 

 

# question= st.chat_input("Enter your question here")

# st.write(chain.invoke({"question":question}))



