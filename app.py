import streamlit as st 
import langchain
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.embeddings import  OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv()

# Get The Pdf Text
def get_pdf_text(pdf_docs):
    text = ""
    # for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def  text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_Store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(user_question)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write( message.content)
        else:
            st.write( message.content)



def main():
    st.set_page_config(page_title='Book Analysis')

    st.header('Books Analyse')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a Question About Your Book:")
    if user_question:
        handle_userinput(user_question)



    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs=st.file_uploader("Slect Your file")
        if st.button("Process"):
            with st.spinner("processing"):
                # Get The Pdf Text
                row_text =get_pdf_text(pdf_docs)

                # Get the Text chunks
                text_chunk=text_chunks(row_text)
                # st.write(len(text_chunk))

                # Create vector database/Store
                vector_store=get_vector_Store(text_chunk)
                
                # create conversation chain
                
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__=='__main__':
    main()