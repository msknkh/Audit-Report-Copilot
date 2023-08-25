import os
import dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document
# Load and preprocess the PDF document
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader

# Load environment variables from .env file
load_dotenv()

# Create an instance of the AzureChatOpenAI class using Azure OpenAI
def create_llm():
    return  AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
        temperature=0.7,
        openai_api_version=os.getenv("OPENAI_DEPLOYMENT_VERSION"))

def create_embeddings():
    return OpenAIEmbeddings(
        openai_api_type="azure",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        deployment=os.getenv("OPENAI_DEPLOYMENT_EMBEDDING_NAME"),
        model="text-embedding-ada-002",
        chunk_size=1)

#loader = PyPDFLoader('/content/INDAS1.pdf')
#documents = loader.load()

load_dotenv()  # Load variables from .env file


def load_documents(directory):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def split_documents_into_chunks(docs, chunk_size, chunk_overlap):

    chunked_docs = []
    for doc in docs:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata.get("page", 1),
                    "chunk": i + 1,
                    "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                },
            )
            chunked_docs.append(doc)

    return chunked_docs

    #text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #return text_splitter.split_documents(docs)


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
    llm = create_llm()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = create_qa_chain(llm, docsearch.as_retriever(), memory=memory)

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
