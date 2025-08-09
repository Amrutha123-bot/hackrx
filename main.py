import streamlit as st
import os
import pickle
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import shutil
from pdf2image import convert_from_path
# --- Langchain Imports ---
# Document Loaders
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.email import UnstructuredEmailLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import unstructured
# Embeddings and LLMs
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
# Vector Stores
from langchain_community.vectorstores import Chroma
# Removed FAISS import as we are using Chroma

# Chains and Prompts
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import SelfQueryRetriever
from langchain.output_parsers import PydanticOutputParser
import os
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"
# Correctly import Document
from langchain_core.documents import Document
# Note: AttributeInfo is not directly used for defining metadata structure; use dictionaries.

# --- Pydantic Models for Structured Data ---
class QueryDetails(BaseModel):
    """Details extracted from a user query."""
    age: Optional[int] = Field(None, description="Age of the person.")
    procedure: Optional[str] = Field(None, description="Medical procedure or service.")
    location: Optional[str] = Field(None, description="Location related to the query (e.g., city, hospital).")
    policy_duration: Optional[str] = Field(None, description="Duration of the insurance policy (e.g., '3 months', '1 year').")

class DecisionResponse(BaseModel):
    """Structured response for decision making."""
    decision: str = Field(..., description="The final decision (e.g., 'approved', 'rejected', 'pending').")
    amount: Optional[float] = Field(None, description="Payout amount, if applicable.")
    justification: str = Field(..., description="Explanation for the decision, referencing specific clauses.")
    clauses_used: List[str] = Field(..., description="List of document clauses that supported the decision.")

# --- Configuration ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set the OPENAI_API_KEY environment variable in your .env file or system settings.")
    st.stop()

# Constants
TEMP_DIR = "uploaded_docs"
CHROMA_DB_DIR = "chroma_db" # Directory for Chroma DB

# --- Initialize Session State ---
def initialize_session_state():
    st.session_state.setdefault('processed_data_path', None) # Will store the Chroma DB directory path
    st.session_state.setdefault('retriever', None)
    st.session_state.setdefault('vectorstore_and_retriever_loaded', False)
    st.session_state.setdefault('llm_parser_decision', None)
    st.session_state.setdefault('llm_retriever', None)
    st.session_state.setdefault('embeddings', None)
    st.session_state.setdefault('status_message', "") # For custom status updates

initialize_session_state()

# --- Initialize LLMs and Embeddings ---
try:
    # Use a slightly more robust model for reasoning if possible, but gpt-3.5-turbo is fine for many tasks
    if st.session_state.llm_parser_decision is None:
        st.session_state.llm_parser_decision = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    if st.session_state.llm_retriever is None:
        st.session_state.llm_retriever = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    if st.session_state.embeddings is None:
        st.session_state.embeddings = OpenAIEmbeddings()
except Exception as e:
    st.error(f"Error initializing OpenAI models. Ensure your OPENAI_API_KEY is set correctly and you have installed the necessary libraries (e.g., `pip install langchain-openai`). Error: {e}")
    st.stop()

# --- Utility Functions ---

def get_loader_for_file(file_path: str):
    """Returns the appropriate LangChain document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        return UnstructuredPDFLoader(file_path)
    elif file_extension == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif file_extension == ".eml":
        return UnstructuredEmailLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path)
    else:
        try:
            # Fallback for other types if langchain-unstructured is installed
            
            st.warning(f"Attempting to load unknown file type '{file_extension}' with UnstructuredLoader.")
            from langchain.document_loaders import UnstructuredFileLoader

            loader = UnstructuredFileLoader(file_path, mode="elements")
            docs = loader.load()
            return docs
        except ImportError:
            st.sidebar.error("`langchain-unstructured` not installed. Cannot load unknown file types. Please install it: `pip install unstructured[all]`")
            return None
        except Exception as e:
            st.sidebar.error(f"Error initializing UnstructuredLoader for {file_path}: {e}")
            return None


def load_documents_from_directory_manually(directory_path):
    """Loads documents from a directory by iterating through files and using specific loaders."""
    all_docs = []
    files_to_process = []

    if not os.path.exists(directory_path):
        st.error(f"Directory not found: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            files_to_process.append(file_path)

    total_files = len(files_to_process)
    if total_files == 0:
        st.warning("No files found in the upload directory.")
        return []

    progress_bar_placeholder = st.empty()
    processed_file_count = 0

    for i, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        try:
            loader = get_loader_for_file(file_path)
            if loader:
                docs = loader.load()
                if not docs:
                    st.sidebar.warning(f"Loader for {filename} returned empty. Check file content or dependencies.")
                else:
                    for doc in docs:
                        if doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata["source"] = filename # Store original filename as source
                        # --- IMPORTANT: Metadata Extraction ---
                        # For SelfQueryRetriever to filter by 'age', 'procedure', etc.,
                        # these fields MUST be populated in doc.metadata here.
                        # This requires custom logic to parse the document's content
                        # and extract these specific fields. The current code only
                        # adds 'source'. You'll need to implement extraction logic
                        # for your specific document formats if you want metadata filtering.
                    all_docs.extend(docs)
                    processed_file_count += 1
            else:
                st.sidebar.warning(f"Skipping unsupported file type: {filename}")

            progress_bar_placeholder.progress((i + 1) / total_files, text=f"Processing: {filename}")

        except Exception as e:
            st.sidebar.error(f"Error loading {filename}: {e}")

    progress_bar_placeholder.empty()

    if not all_docs and files_to_process:
        st.error("Failed to load any documents. Please check file formats and ensure necessary dependencies are installed (e.g., pip install unstructured[pdf] python-docx pypdf).")
    elif not all_docs and not files_to_process:
        st.warning("No files were found to process.")
    else:
        st.success(f"Successfully loaded {len(all_docs)} document chunks from {processed_file_count} files.")

    return all_docs

def parse_user_query(query: str) -> QueryDetails:
    """Parses a natural language query into structured details using Pydantic."""
    if not query:
        return QueryDetails()

    parser = PydanticOutputParser(pydantic_object=QueryDetails)

    prompt_template = """
    You are an expert at extracting specific information from user queries related to insurance policies.
    Parse the following query and return the extracted details in a JSON format.
    Identify age, medical procedures, locations, and policy durations.

    Query: "{user_query}"

    {format_instructions}
    """

    formatted_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["user_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        response = st.session_state.llm_parser_decision.invoke(formatted_prompt.format(user_query=query))
        parsed_data = parser.parse(response.content)
        return parsed_data
    except Exception as e:
        st.error(f"Error parsing query: {e}")
        # Return empty QueryDetails to avoid breaking the flow if parsing fails
        return QueryDetails()

def create_decision_chain(llm, retriever):
    """Creates a chain for making decisions based on retrieved documents."""
    decision_prompt_template = """
    You are an expert in policy analysis and claim processing.
    Based on the following retrieved document clauses and the user's original query,
    make a decision about the claim and provide a clear justification.
    If a payout is applicable, specify the amount.
    List all clauses used to reach the decision.

    User Query: "{user_query}"

    Retrieved Clauses:
    {context}

    Please provide your response in the following JSON format:
    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=DecisionResponse)

    prompt = PromptTemplate(
        template=decision_prompt_template,
        input_variables=["user_query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain, parser

# --- Streamlit UI ---

st.set_page_config(page_title="AI Policy Assistant", layout="wide")
st.title("AI Policy Assistant 🤖📄")
st.markdown("Upload your policy documents (PDF, DOCX, EML, TXT) and ask questions.")

# Sidebar for uploads and processing
st.sidebar.title("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents",
    type=["pdf", "docx", "eml", "txt"],
    accept_multiple_files=True
)

process_button = st.sidebar.button("Process Documents", key="process_button")
clear_cache_button = st.sidebar.button("Clear Processed Data", key="clear_cache_button")

# Placeholder for status messages in the main area
status_placeholder = st.empty()

# --- File Processing Logic ---
if process_button:
    if uploaded_files:
        # --- 1. Save Uploaded Files ---
        os.makedirs(TEMP_DIR, exist_ok=True)
        files_saved_count = 0
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                files_saved_count += 1
            except Exception as e:
                st.sidebar.error(f"Error saving file {uploaded_file.name}: {e}")
                st.session_state.status_message = f"Error saving file {uploaded_file.name}"

        if files_saved_count == 0:
            st.session_state.status_message = "No files were successfully saved."
        else:
            st.session_state.status_message = f"Saved {files_saved_count} files locally. Starting document processing..."
            status_placeholder.info(st.session_state.status_message)

            # --- 2. Load Documents ---
            all_docs = load_documents_from_directory_manually(TEMP_DIR)

            if not all_docs:
                st.session_state.status_message = "Document loading failed. Please check file formats and dependencies."
                status_placeholder.error(st.session_state.status_message)
            else:
                # --- 3. Split Documents ---
                st.session_state.status_message = "Splitting documents into chunks..."
                status_placeholder.info(st.session_state.status_message)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                split_docs = splitter.split_documents(all_docs)

                if not split_docs:
                    st.session_state.status_message = "Failed to split documents. Content might be too short or malformed."
                    status_placeholder.error(st.session_state.status_message)
                else:
                    # --- 4. Create Embeddings and Vector Store (using Chroma) ---
                    st.session_state.status_message = "Creating embeddings and indexing documents into Chroma DB..."
                    status_placeholder.info(st.session_state.status_message)

                    # Define metadata fields for SelfQueryRetriever.
                    # IMPORTANT: For filtering to work, these fields MUST be populated in the `doc.metadata`
                    # when the documents are loaded and indexed. You need to implement logic to extract
                    # 'age', 'procedure', 'location', 'policy_duration' from document content if required for filtering.
                    metadata_field_info = [
                        {"name": "source", "description": "The source document the chunk came from", "type": "string"},
                        # Add other fields here IF you have logic to populate them during doc loading:
                        {"name": "age", "description": "The age of the policyholder.", "type": "integer"},
                        {"name": "procedure", "description": "The medical procedure performed or service requested.", "type": "string"},
                        {"name": "location", "description": "The location of the service or hospital.", "type": "string"},
                        {"name": "policy_duration", "description": "The duration of the insurance policy.", "type": "string"},
                    ]

                    try:
                        # ChromaDB will create the directory if it doesn't exist during from_documents
                        vectorstore = Chroma.from_documents(
                            documents=split_docs,
                            embedding=st.session_state.embeddings,
                            persist_directory=CHROMA_DB_DIR # Specify directory for Chroma
                        )
                        # No explicit .persist() needed for recent ChromaDB versions with persist_directory

                        st.session_state.processed_data_path = CHROMA_DB_DIR # Store the Chroma DB directory path
                        # Save the vectorstore object itself into session state as well if needed for other operations
                        st.session_state.vectorstore = vectorstore # Store the vectorstore object

                        # --- 5. Create Retriever ---
                        st.session_state.retriever = SelfQueryRetriever.from_llm(
                            st.session_state.llm_retriever, # LLM for query understanding
                            vectorstore,
                            # Description of the documents for the retriever
                            "A collection of policy documents and related information.",
                            metadata_field_info, # Metadata definitions for filtering
                            verbose=True,
                            search_kwargs={'k': 5} # Number of documents to retrieve
                        )
                        st.session_state.vectorstore_and_retriever_loaded = True

                        st.session_state.status_message = "Documents processed and indexed successfully! You can now ask questions."
                        status_placeholder.success(st.session_state.status_message)

                    except Exception as e:
                        st.session_state.status_message = f"Error during embedding or Chroma DB creation: {e}"
                        status_placeholder.error(st.session_state.status_message)
                        # Clean up potentially partially created directory if error occurs
                        if os.path.exists(CHROMA_DB_DIR):
                            try:
                                shutil.rmtree(CHROMA_DB_DIR)
                                st.sidebar.warning(f"Cleaned up incomplete Chroma DB directory: '{CHROMA_DB_DIR}' due to error.")
                            except Exception as rm_e:
                                st.sidebar.error(f"Could not remove incomplete Chroma DB directory: {rm_e}")
                        st.session_state.processed_data_path = None
                        st.session_state.retriever = None
                        st.session_state.vectorstore_and_retriever_loaded = False
    else:
        st.session_state.status_message = "Please upload at least one document to process."
        status_placeholder.warning(st.session_state.status_message)

# --- Clear Cache Logic ---
if clear_cache_button:
    # Clear uploaded files directory
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            st.session_state.status_message = "Temporary document directory cleared."
        except Exception as e:
            st.sidebar.error(f"Error clearing temporary directory {TEMP_DIR}: {e}")
            st.session_state.status_message = f"Error clearing temp dir: {e}"

    # Clear Chroma DB directory
    chroma_dir_path = st.session_state.get('processed_data_path', CHROMA_DB_DIR) # Fallback to default if state not set
    if chroma_dir_path and os.path.exists(chroma_dir_path):
        try:
            shutil.rmtree(chroma_dir_path)
            st.session_state.status_message = f"Chroma DB directory '{chroma_dir_path}' cleared."
        except Exception as e:
            st.sidebar.error(f"Error clearing Chroma DB directory {chroma_dir_path}: {e}")
            st.session_state.status_message = f"Error clearing Chroma DB: {e}"

    # Reset session state
    st.session_state.processed_data_path = None
    st.session_state.retriever = None
    st.session_state.vectorstore = None # Clear stored vectorstore too
    st.session_state.vectorstore_and_retriever_loaded = False
    status_placeholder.success("Processed data cleared. Please re-upload documents.")
    st.rerun() # Use the correct method to refresh the app

# --- Load Vector Store and Retriever if it exists ---
# This block runs when the app starts or reruns, to load previously processed data.
if (st.session_state.processed_data_path and os.path.exists(st.session_state.processed_data_path)) or \
   (not st.session_state.processed_data_path and os.path.exists(CHROMA_DB_DIR)): # Also check default path if state is missing

    if not st.session_state.vectorstore_and_retriever_loaded: # Only load if not already loaded in this session
        load_path = st.session_state.processed_data_path if st.session_state.processed_data_path else CHROMA_DB_DIR
        status_placeholder.info(f"Loading existing Chroma DB from: {load_path}")
        st.session_state.status_message = f"Loading existing Chroma DB from: {load_path}"

        try:
            # --- Load Chroma DB ---
            # Make sure the embeddings are re-initialized or accessible
            if st.session_state.embeddings is None:
                 st.session_state.embeddings = OpenAIEmbeddings()

            vectorstore = Chroma(
                persist_directory=load_path,
                embedding_function=st.session_state.embeddings # Use embeddings from session state
            )

            # Define metadata_field_info for retriever instantiation
            # This must match the structure used during creation.
            metadata_field_info_for_retriever = [
                {"name": "source", "description": "The source document the chunk came from", "type": "string"},
                # Add other fields if they were populated and you want to filter by them
            ]

            # Create/recreate the retriever if it's not already in session state
            if st.session_state.retriever is None:
                st.session_state.retriever = SelfQueryRetriever.from_llm(
                    st.session_state.llm_retriever,
                    vectorstore,
                    "A collection of policy documents and related information.", # Description
                    metadata_field_info_for_retriever, # Pass the metadata definitions
                    verbose=True,
                    search_kwargs={'k': 5}
                )

            st.session_state.processed_data_path = load_path # Ensure state reflects the loaded path
            st.session_state.vectorstore = vectorstore # Store the loaded vectorstore object
            st.session_state.vectorstore_and_retriever_loaded = True # Mark as loaded
            status_placeholder.success("Chroma DB and retriever loaded successfully.")
            st.session_state.status_message = "Chroma DB and retriever loaded successfully."

        except Exception as e:
            st.session_state.status_message = f"Error loading existing Chroma DB: {e}"
            status_placeholder.error(st.session_state.status_message)
            # Clean up potentially corrupted state or directory
            st.session_state.processed_data_path = None
            st.session_state.retriever = None
            st.session_state.vectorstore = None
            st.session_state.vectorstore_and_retriever_loaded = False
            # Optionally remove the corrupted DB directory
            if load_path and os.path.exists(load_path):
                try:
                    shutil.rmtree(load_path)
                    st.sidebar.info(f"Cleaned up potentially corrupted Chroma DB directory: '{load_path}'")
                except Exception as rm_e:
                    st.sidebar.error(f"Could not remove corrupt Chroma DB directory: {rm_e}")
            st.rerun() # Rerun to reset the state if loading failed

# --- Query Interface ---
st.header("Ask a Question")

# Only show query input if vectorstore and retriever are ready
if st.session_state.vectorstore_and_retriever_loaded and st.session_state.retriever:
    user_query = st.text_input("Enter your query about the documents:", key="query_input")

    if user_query:
        status_placeholder.info("Processing your query...")
        st.session_state.status_message = "Processing your query..."

        # Optional: Parse query details for potential future use or validation
        # query_details = parse_user_query(user_query)
        # st.write(f"Parsed Query Details: {query_details}") # For debugging

        try:
            # Use the retriever from session state
            # If you want to use the parsed query_details for filtering,
            # you'd pass them here:
            # relevant_docs = st.session_state.retriever.get_relevant_documents(user_query, filter=...)
            relevant_docs = st.session_state.retriever.get_relevant_documents(user_query)

            if not relevant_docs:
                st.warning("No relevant documents found for your query.")
                st.session_state.status_message = "No relevant documents found."
            else:
                # Use the LLM stored in session state for creating the chain
                decision_chain, decision_parser = create_decision_chain(st.session_state.llm_parser_decision, st.session_state.retriever)

                context_for_llm = "\n\n".join([
                    f"--- Document: {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}"
                    for doc in relevant_docs
                ])

                try:
                    response_content = decision_chain.invoke({
                        "user_query": user_query,
                        "context": context_for_llm
                    })

                    # Parse the LLM's structured response
                    decision_response = decision_parser.parse(response_content['text'])

                    # Display the results
                    st.subheader("Decision and Justification")
                    st.write(f"*Decision:* **{decision_response.decision}**")
                    if decision_response.amount is not None:
                        st.write(f"*Amount:* **${decision_response.amount:,.2f}**") # Formatted amount
                    st.write(f"*Justification:* {decision_response.justification}")

                    if decision_response.clauses_used:
                        st.subheader("Supporting Clauses:")
                        for clause in decision_response.clauses_used:
                            st.markdown(f"- {clause}")

                    status_placeholder.success("Query processed successfully!")
                    st.session_state.status_message = "Query processed successfully."

                except Exception as e:
                    # Handle errors during decision making or parsing
                    status_placeholder.error(f"Error during decision making or parsing: {e}")
                    st.session_state.status_message = f"Error during decision making: {e}"
                    if 'response_content' in locals() and response_content.get('text'):
                        with st.expander("Show Raw AI Output for Debugging"):
                            st.text(response_content['text'])

        except Exception as e:
            # Handle errors during document retrieval
            st.error(f"Error retrieving documents: {e}")
            st.session_state.status_message = f"Error retrieving documents: {e}"
            status_placeholder.error("Failed to retrieve relevant information.")
else:
    # This message is shown if vectorstore or retriever are not ready
    st.info("Please upload and process your documents first to enable querying.")
    st.session_state.status_message = "Please upload and process documents."

# Display status message in the designated placeholder
status_placeholder.info(st.session_state.status_message)