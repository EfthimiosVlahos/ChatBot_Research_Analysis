import os
import pickle
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Obtain API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If the API key could not be obtained from environment variables, try to get it from Streamlit secrets
if OPENAI_API_KEY is None:
    OPENAI_API_KEY = st.secrets["my_secrets"]["OPENAI_API_KEY"]

# Validate API Key
if OPENAI_API_KEY is None:
    raise ValueError("API Key not found. Ensure your .env file or Streamlit secrets contain 'OPENAI_API_KEY'.")

# Set the title of the Streamlit application
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Gather up to 3 URLs from the user for news articles to process
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

# Button to start processing the URLs
process_url_clicked = st.sidebar.button("Process URLs")

# File path for saving the FAISS index
file_path = "faiss_store_openai.pkl"

# Initializing progress bar
main_placeholder = st.empty()

# Initialize the OpenAI model with specific parameters
try:
    llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI model: {str(e)}")
    raise

# # Start processing when the button is clicked
# if process_url_clicked:
#     try:
#         # Load data from the provided URLs
#         loader = UnstructuredURLLoader(urls=urls)
#         main_placeholder.text("Data Loading...Started...✅✅✅")
#         data = loader.load()

#         # Split the loaded data into smaller chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", ".", ","], chunk_size=1000
#         )
#         main_placeholder.text("Text Splitter...Started...✅✅✅")
#         docs = text_splitter.split_documents(data)

#         # create embeddings and save it to FAISS index
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         vectorstore_openai = FAISS.from_documents(docs, embeddings)
#         main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#         time.sleep(2)

#         # Save the FAISS index to a pickle file for later retrieval
#         with open(file_path, "wb") as f:
#             pickle.dump(vectorstore_openai, f)
#     except Exception as e:
#         st.error(f"An error occurred while processing the URLs: {str(e)}")
#         raise
    
    
if process_url_clicked:
    try:
        # Load data from the provided URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split the loaded data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Validate if 'docs' is non-empty and valid
        if not docs:
            raise ValueError("Docs are empty or invalid.")

        # Generate embeddings for the split documents and store them in a FAISS index
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Validate if 'embeddings' are non-empty and valid
        sample_embedding = embeddings.encode(["Sample text"])
        if not sample_embedding:
            raise ValueError("Embeddings are empty or invalid.")
        
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file for later retrieval
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
    except Exception as e:
        st.error(f"An error occurred while processing the URLs: {str(e)}")
        raise

# Allow the user to input a question to query the processed documents
query = main_placeholder.text_input("Question: ")

# Process the query and provide an answer if a valid question is entered
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                # Load the saved FAISS index
                vectorstore = pickle.load(f)

                # Use the OpenAI model and FAISS index to retrieve an answer for the query
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)

                # Display the answer
                st.header("Answer")
                st.write(result["answer"])

                # Display the sources (documents) from which the answer was derived
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.error(f"An error occurred while retrieving the answer: {str(e)}")
            raise
