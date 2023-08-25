import openai
import streamlit as st
from langchain.llms import OpenAI

st.title("Your Auditor")

# Create a sidebar with a file uploader
st.sidebar.title("PDF Uploader")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Check if a file is uploaded
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai.api_key)
    st.info(llm(input_text))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})