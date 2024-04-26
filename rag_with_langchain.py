# pip install --upgrade --quiet langchain langchain-google-vertexai langchain-chroma

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import TextLoader

# Initialize the Google VertexAI model
# "som-rit-phi-starr-dev", "us-west1"
llm = ChatVertexAI(model="gemini-1.5-pro-preview-0409")

# Initialize the VertexAIEmbeddings class
# "som-rit-phi-starr-dev", "us-west1"
embeddings = VertexAIEmbeddings("textembedding-gecko")
# ans = embeddings.embed_query("What is the weather in New York?")
# print(ans)

# load your document
loader = TextLoader("stetson-help.txt")
docs = loader.load()
# print(docs)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500
)
all_splits = text_splitter.split_documents(docs)

print("Total Chunks " + str(len(all_splits)))

# Create a Chroma (vector store) object from the documents and embeddings
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum and keep the answer as concise as possible.
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, use your own knowledge to provide a helpful response. 
Keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(inp_docs):
    str = "\n\n".join(doc.page_content for doc in inp_docs)
    # print("\nContext Information: " + "----------\n\n".join(doc.page_content for doc in inp_docs))
    return str


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
)

# result = rag_chain.invoke("My IRB is not visible in the system. Can you help me with that?")
# print(result)
print("Welcome to Stetson AI Chatbot. Type 'exit' to quit.\n")

while True:
    question = input("Enter your question: ")
    if question == "exit":
        break
    result = rag_chain.invoke(question)
    print("Answer: " + result)

vectorstore.delete_collection()

