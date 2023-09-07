import streamlit as st
import time
import numpy as np
import base64
import os
import tempfile

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, Replicate
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# Load the OPENAI_API_KEY from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(
    page_title="Main Page",
    page_icon="👋"
)

def create_embeddings():
    # return OpenAIEmbeddings(
    #     openai_api_type="azure",
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     openai_api_base=os.getenv("OPENAI_API_BASE"),
    #     openai_api_version=os.getenv("OPENAI_API_BASE"),
    #     deployment=os.getenv("OPENAI_DEPLOYMENT_VERSION"),
    #     model="text-embedding-ada-002",
    #     chunk_size=1)
    return HuggingFaceEmbeddings()

# def create_embeddings():
#     return OpenAIEmbeddings()


def create_chroma_from_documents(texts, embeddings):
    # if os.path.exists("../chroma_db"):
    #     # If the directory exists, load the existing Chroma instance
    #     return Chroma(persist_directory="../chroma_db", embedding_function=embeddings)
    # else:
        # If the directory doesn't exist, create a new Chroma instance and save it to disk
    return Chroma.from_documents(texts, embeddings)


# @st.cache_resource(ttl='1hr')
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        

        # Create embeddings and store in vectordb
        embeddings = create_embeddings();
        vectordb = Chroma.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

        return retriever


def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        container.write("**Context Retrieval**")
        self.status = container

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


with st.sidebar:
    st.header("Projects")
    st.button("Add new project", type="secondary")
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.")
#     st.stop()


retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
# llm = OpenAI(
#     model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
# )
# Initialize Replicate Llama2 Model
llm = Replicate(
    model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"temperature": 0.75, "max_length": 3000}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory
)


st.markdown("#Markdown demo")

st.divider()

col1, col2 = st.columns(2)


msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}

user_query = st.chat_input(placeholder="Ask me anything!")
    
with col1:
    st.header("Chat")
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query: 
        st.chat_message("user").write(user_query)
        msgs.add_user_message(user_query)
     
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query , callbacks=[retrieval_handler, stream_handler])

        st.chat_message("assistant").write(response)
        msgs.add_ai_message(response)

with col2:
    show_pdf("/Users/muskankhandelwal/auditGPT/accounting_standard_pdfs/INDAS2.pdf")
