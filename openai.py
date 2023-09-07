from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.schema import SystemMessage
from langchain.document_loaders import PyPDFLoader
import os

load_dotenv()  # Load variables from .env file

# Load the OPENAI_API_KEY from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


def load_documents(directory):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def split_documents_into_chunks(docs, chunk_size, chunk_overlap):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_embeddings():
    return OpenAIEmbeddings()

def create_chroma_from_documents(texts, embeddings):
    if os.path.exists("./chroma_db"):
        # If the directory exists, load the existing Chroma instance
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        # If the directory doesn't exist, create a new Chroma instance and save it to disk
        return Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

def create_qa_chain(llm, retriever, memory):
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

def add_new_pdf():
    # Load and preprocess the PDF document
    loader = PyPDFLoader('accounting_standard_pdfs/INDAS2.pdf')
    documents = loader.load()

    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    page_content_list = [doc.page_content for doc in texts]

    vectorstore.add_texts(page_content_list)

def main():
    directory = "accounting_standard_pdfs"
    chunk_size = 1000
    chunk_overlap = 0

    docs = load_documents(directory)
    texts = split_documents_into_chunks(docs, chunk_size, chunk_overlap)

    embeddings = create_embeddings()
    docsearch = create_chroma_from_documents(texts, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = create_qa_chain(OpenAI(temperature=0), docsearch.as_retriever(), memory=memory)

    #add_new_pdf()

    queries = [
        "What are statement of cash flow",
        "Difference between INDAS7 and INDAS2"
    ]

    for query in queries:
        result = qa({"question": query})
        print(result['answer'])

if __name__ == "__main__":
    main()
