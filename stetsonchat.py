import streamlit as st
from streamlit_chat import message
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader

# Initialize the Google VertexAI model
llm = ChatVertexAI(model="gemini-1.5-pro-preview-0409")

# Initialize the VertexAIEmbeddings class
embeddings = VertexAIEmbeddings("textembedding-gecko")

# Load your document
loader = TextLoader("stetson-help.txt")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
all_splits = text_splitter.split_documents(docs)

# Create a Chroma (vector store) object from the documents and embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define a custom prompt template for responses
template = """
Use the provided context about Stetson to answer the question at the end.
If the question is unrelated to Stetson, please ask for clarification or 
suggest a more relevant question about Stetson. If you don't know the answer 
but it's related to Stetson, use your knowledge to provide the best possible response. 
Aim for concise answers and always conclude with 'Thanks for asking!

{context}

Question: {question}

Helpful Answer:
"""

custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(inp_docs):
    return "\n\n".join(doc.page_content for doc in inp_docs)

# Create the processing chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("Stetson AI Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def handle_message():
    question = st.session_state.user_input
    if question:
        st.session_state.messages.append({'message': question, 'is_user': True})
        try:
            result = rag_chain.invoke(question)
            st.session_state.messages.append({'message': "Answer: " + result, 'is_user': False})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def clear_messages():
    st.session_state.messages = []

chat_placeholder = st.empty()

with chat_placeholder.container():
    for idx, chat in enumerate(st.session_state.messages):
        message(chat['message'], is_user=chat['is_user'], key=f"chat_{idx}")

    st.text_input("User Input:", key="user_input", on_change=handle_message)
    st.button("Clear Messages", on_click=clear_messages)
