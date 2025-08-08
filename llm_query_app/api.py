from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import os
import shutil
import time
import requests
from dotenv import load_dotenv
import json # For JSON parsing of LLM output

# --- Langchain Imports ---
# Ensure these are installed:
# pip install fastapi uvicorn python-dotenv requests langchain-core langchain-community langchain-openai langchain-chroma unstructured[pdf] pdfminer.six python-docx pydantic
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.base import RunnableParallel # For parallel processing if needed later


# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")

if not OPENAI_API_KEY:
    # This will cause the app to fail on startup if the key is not set
    raise RuntimeError("OPENAI_API_KEY not found. Please set the environment variable.")
if not HACKRX_API_KEY:
    # This will cause the app to fail on startup if the key is not set
    raise RuntimeError("HACKRX_API_KEY not found. Please set the environment variable.")


# --- Constants ---
CHROMA_DB_DIR = "./chroma_db_api" # Directory for ChromaDB persistence
COLLECTION_NAME = "policy_docs_api"
MAX_RETRIES_LLM_OUTPUT = 3 # Number of times to retry LLM output if validation fails
DOCUMENT_DOWNLOAD_TIMEOUT = 30 # Timeout for downloading documents

# --- Pydantic Models ---
# These are used for internal processing and structuring LLM outputs.
class QueryDetails(BaseModel):
    """Model for parsing user's natural language query into structured details."""
    age: Optional[int] = Field(None, description="Age of the person.")
    procedure: Optional[str] = Field(None, description="Medical procedure or service.")
    location: Optional[str] = Field(None, description="Location related to the query (e.g., city, hospital).")
    policy_duration: Optional[str] = Field(None, description="Duration of the insurance policy (e.g., '3 months', '1 year').")

class DecisionResponse(BaseModel):
    """Model for the final decision, amount, justification, and supporting clauses."""
    decision: str = Field(..., description="The final decision (e.g., 'approved', 'rejected', 'pending').")
    amount: Optional[float] = Field(None, description="Payout amount, if applicable.")
    justification: str = Field(..., description="Explanation for the decision, referencing specific clauses or policy terms.")
    clauses_used: List[Dict[str, str]] = Field(..., description="List of dictionaries, where each dict contains 'clause_text' and 'source_document'.")

# --- API Request/Response Models (HackRx Platform Format) ---
class HackRxRequest(BaseModel):
    documents: Optional[str] = Field(None, description="URL to the document. Required if vector store is not initialized.")
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Metadata Schema for SelfQueryRetriever ---
# Ensure this is accessible by LangchainManager and relevant functions
METADATA_FIELD_INFO = [
    AttributeInfo(name="source", description="The source document the chunk came from", type="string"),
    AttributeInfo(name="page", description="The page number the chunk came from", type="integer"), # Added page number if available
    AttributeInfo(name="age", description="The age of the policyholder.", type="integer"),
    AttributeInfo(name="procedure", description="The medical procedure performed or service requested.", type="string"),
    AttributeInfo(name="location", description="The location of the service or hospital.", type="string"),
    AttributeInfo(name="policy_duration", description="The duration of the insurance policy.", type="string"),
]

# --- Langchain Manager (Singleton Pattern for efficiency) ---
class LangchainManager:
    def __init__(self):
        print("Initializing LangchainManager...")
        self.embeddings = OpenAIEmbeddings()
        # Use models suitable for speed and accuracy for hackathon tasks
        self.llm_parser = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # For parsing structured data
        self.llm_retriever_config = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1) # For retriever config
        self.llm_decision_maker = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2) # For making decisions

        self.vectorstore = None
        self.retriever = None
        self.vectorstore_loaded = False
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """
        Loads existing vector store or attempts to create a new one.
        This method is called upon initialization.
        """
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR): # Check if dir exists and is not empty
            print(f"Attempting to load existing vector store from '{CHROMA_DB_DIR}'...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )
                self.retriever = SelfQueryRetriever.from_llm(
                    self.llm_retriever_config,
                    self.vectorstore,
                    "Documents containing insurance policy details, claims, and medical procedures.",
                    METADATA_FIELD_INFO,
                    verbose=True,
                    search_kwargs={'k': 5} # Adjust k based on desired recall/precision balance
                )
                self.vectorstore_loaded = True
                print("Successfully loaded existing vector store.")
            except Exception as e:
                print(f"Error loading vector store from '{CHROMA_DB_DIR}': {e}. Attempting to recreate...")
                self._create_vectorstore() # Attempt to create if loading fails
        else:
            print("No existing vector store found. Attempting to create a new one (requires 'policy.pdf' or URL processing).")
            self._create_vectorstore() # Create if directory doesn't exist or is empty

    def _create_vectorstore(self, document_url: Optional[str] = None):
        """
        Creates and persists a new vector store.
        Can be initialized with a default document or a provided URL.
        """
        docs_to_process = []
        source_document_name = "default_policy.pdf" # Default if no URL is given

        if document_url:
            print(f"Processing document from provided URL: {document_url}")
            try:
                response = requests.get(document_url, timeout=DOCUMENT_DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                temp_doc_path = "temp_policy_for_api.pdf"
                with open(temp_doc_path, "wb") as f:
                    f.write(response.content)
                
                loader = UnstructuredPDFLoader(temp_doc_path)
                docs_to_process = loader.load()
                source_document_name = os.path.basename(document_url) if document_url else "temp_policy_for_api.pdf"

                if not docs_to_process:
                    raise ValueError("Document loaded but contains no content.")

            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to download document from URL '{document_url}': {e}")
            except Exception as e:
                raise RuntimeError(f"Error processing document from URL '{document_url}': {e}")
            finally:
                if 'temp_doc_path' in locals() and os.path.exists(temp_doc_path):
                    os.remove(temp_doc_path) # Clean up temporary file

        elif os.path.exists("policy.pdf"): # Fallback to a local policy.pdf if no URL provided
            print("No URL provided, attempting to use local 'policy.pdf' for initial setup...")
            try:
                loader = UnstructuredPDFLoader("policy.pdf")
                docs_to_process = loader.load()
                source_document_name = "policy.pdf"
                if not docs_to_process:
                    raise ValueError("Local 'policy.pdf' loaded but contains no content.")
            except Exception as e:
                raise RuntimeError(f"Error processing local 'policy.pdf': {e}")
        else:
            print("No document URL provided and no local 'policy.pdf' found. Cannot create initial vector store.")
            return # Do not proceed if no documents are available

        # --- Document Splitting ---
        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = splitter.split_documents(docs_to_process)

        if not split_docs:
            raise RuntimeError("Failed to create document chunks after splitting.")
        
        print(f"Created {len(split_docs)} document chunks.")

        # --- ChromaDB Setup ---
        print(f"Creating and persisting new vector store to '{CHROMA_DB_DIR}'...")
        # Clean up existing DB if it exists to ensure fresh index
        if os.path.exists(CHROMA_DB_DIR):
            print(f"Clearing existing ChromaDB at {CHROMA_DB_DIR}...")
            shutil.rmtree(CHROMA_DB_DIR)
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

        self.vectorstore = Chroma.from_documents(
            split_docs,
            self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_DIR
        )
        self.retriever = SelfQueryRetriever.from_llm(
            self.llm_retriever_config,
            self.vectorstore,
            "Documents containing insurance policy details, claims, and medical procedures.",
            METADATA_FIELD_INFO,
            verbose=True,
            search_kwargs={'k': 5}
        )
        self.vectorstore_loaded = True
        print("Successfully created and persisted new vector store.")

    def update_vectorstore_with_url(self, document_url: str):
        """
        Downloads, processes, and updates the vector store with a new document from a URL.
        Clears the old store and creates a new one.
        """
        print(f"Updating vector store with document from URL: {document_url}")
        self._create_vectorstore(document_url=document_url) # Recreates the store

# --- Instance Creation ---
# The instance is created immediately after the class definition.
# This is the standard Python way to create a global instance from a class defined in the same file.
langchain_manager = LangchainManager()
print("LangchainManager instance created.")

# --- Helper function for LLM-based Query Parsing ---
# Used internally by SelfQueryRetriever to understand query metadata.
def parse_query_for_retriever(query: str) -> QueryDetails:
    """
    Parses a natural language query into structured QueryDetails using an LLM.
    Designed for use with SelfQueryRetriever, aims for strict Pydantic V2 validation.
    """
    if not query:
        return QueryDetails() # Return empty object if query is empty

    parser = PydanticOutputParser(pydantic_object=QueryDetails)
    prompt_template = """
    You are an expert at extracting specific information from user queries related to insurance policies.
    Parse the following query and return the extracted details in a JSON format that strictly matches the Pydantic model.
    Identify age, medical procedures, locations, and policy durations. Ensure all fields are correctly typed.
    If a field is not present in the query, return null for that field.

    Query: "{user_query}"

    Strict JSON Output Format:
    {format_instructions}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    for attempt in range(MAX_RETRIES_LLM_OUTPUT):
        try:
            response = langchain_manager.llm_parser.invoke(prompt.format(user_query=query))
            llm_output = response.content

            # Attempt to parse using Pydantic V2's model_validate
            try:
                # LLMs sometimes output raw JSON strings, try to load it
                data = json.loads(llm_output)
                parsed_data = QueryDetails.model_validate(data)
                return parsed_data
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"API Attempt {attempt+1}: Failed to parse LLM output for query parsing. Error: {e}. LLM Output: `{llm_output}`")
                # If direct JSON parsing fails, try the Pydantic parser which might be more lenient
                try:
                    parsed_data = parser.parse(llm_output)
                    return parsed_data
                except ValidationError as ve:
                    print(f"API Attempt {attempt+1}: Validation error even with parser.parse: {ve}. LLM Output: `{llm_output}`")
                    # Continue to next attempt if validation still fails
                    if attempt == MAX_RETRIES_LLM_OUTPUT - 1:
                        print("Max retries reached for query parsing.")
                        return QueryDetails() # Return default if all retries fail
        except Exception as e:
            print(f"API Attempt {attempt+1}: Error during LLM invocation for query parsing: {e}")
            if attempt == MAX_RETRIES_LLM_OUTPUT - 1:
                print("Max retries reached for query parsing LLM invocation.")
                return QueryDetails() # Return default if all retries fail
            time.sleep(1) # Small delay before retrying

    print("Failed to parse user query after multiple attempts. Returning default QueryDetails.")
    return QueryDetails() # Return default if all attempts failed


# --- Helper function to create the decision-making chain ---
def create_decision_chain(llm, output_parser: PydanticOutputParser) -> Runnable:
    """
    Creates a LangChain Expression Language (LCEL) chain for making decisions.
    The LLM takes the user query, retrieved context, and format instructions.
    """
    decision_prompt_template = """
    You are an expert in policy analysis and claim processing.
    Based on the following retrieved document clauses and the user's original query,
    make a decision about the claim and provide a clear justification.
    If a payout is applicable, specify the amount.
    List all clauses used to reach the decision, and for each clause, **also state the source document it came from**.
    Ensure your output strictly adheres to the JSON format requested by the parser.
    If you cannot make a decision or find sufficient information, state "pending" for decision and provide a reason in justification.

    User Query: "{user_query}"

    Retrieved Clauses:
    {context}

    Please provide your response in the following JSON format:
    {format_instructions}
    """
    prompt_template_obj = PromptTemplate.from_template(decision_prompt_template)
    # Use the dedicated decision LLM
    chain: RunnableSequence = prompt_template_obj | llm | output_parser
    return chain

# --- FastAPI Application Instance ---
app = FastAPI(
    title="HackRx AI Policy Assistant API",
    description="API for processing policy documents and answering questions.",
    version="1.0.0",
)

# --- Custom Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors for incoming requests."""
    print(f"Request validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "message": "Invalid input data provided. Please check the payload against the API schema."
        },
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handles critical runtime errors (e.g., API key missing, DB init failed)."""
    print(f"Runtime error occurred: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # Use 503 for service not ready
        content={"detail": str(exc), "message": "Service is not available due to a configuration or initialization error."}
    )

# --- Authentication Dependency ---
def verify_api_key(api_key: str = Depends(lambda request: request.headers.get("Authorization"))):
    """
    Dependency to verify the API key provided in the Authorization header.
    Raises HTTPException if the key is missing, malformed, or invalid.
    """
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing or malformed. Expected 'Bearer <api_key>'."
        )

    token = api_key.split(" ")[1]
    if token != HACKRX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )
    return True # Indicates successful authentication

# --- Document Processing Helper ---
def process_and_update_vectorstore(document_url: str):
    """
    Downloads, processes, and updates the vector store with a new document.
    This function is called when the vector store is not loaded or needs updating.
    """
    print(f"Initiating process for document URL: {document_url}")
    # Pass the URL to the LangchainManager to handle the creation/update
    try:
        langchain_manager._create_vectorstore(document_url=document_url)
        print("Vector store updated successfully.")
    except RuntimeError as e:
        # Catch errors from _create_vectorstore and re-raise as HTTPExceptions
        print(f"Error during vector store update: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process document: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during vector store update: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected server error occurred during document processing.")


# --- API Endpoint for HackRx Submission ---
@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_api_key)])
async def run_hackrx(request_data: HackRxRequest):
    """
    Receives document URL and questions, processes them, and returns answers.
    Handles dynamic document loading if vector store is not initialized.
    """
    start_time = time.time()

    # --- Step 1: Ensure Vector Store is Loaded ---
    if not langchain_manager.vectorstore_loaded:
        if not request_data.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL is required when the vector store is not initialized."
            )
        # Dynamically process and update the vector store
        process_and_update_vectorstore(request_data.documents)

    # --- Step 2: Process Each Question ---
    answers = []
    
    for question in request_data.questions:
        try:
            print(f"\nProcessing question: '{question}'")
            # --- 2a: Retrieve Relevant Documents ---
            # Use the retriever configured with self-query capabilities
            retrieved_docs_for_question = langchain_manager.retriever.invoke({"query": question})

            if not retrieved_docs_for_question:
                print(f"No relevant documents found for question: '{question}'")
                answers.append("Could not find relevant information.")
                continue

            # --- 2b: Prepare Context for LLM ---
            context_for_llm = "\n\n".join([
                f"--- Source: {doc.metadata.get('source', 'Unknown Source')} (Page: {doc.metadata.get('page', 'N/A')}) ---\n{doc.page_content}"
                for doc in retrieved_docs_for_question
            ])
            print(f"Retrieved {len(retrieved_docs_for_question)} chunks for question.")

            # --- 2c: Create and Invoke Decision Chain ---
            decision_parser = PydanticOutputParser(pydantic_object=DecisionResponse)
            decision_chain = create_decision_chain(langchain_manager.llm_decision_maker, decision_parser)

            # Prepare prompt input for the decision chain
            decision_prompt_input = {
                "user_query": question,
                "context": context_for_llm,
                "format_instructions": decision_parser.get_format_instructions()
            }

            llm_raw_response = None # To store the raw output for debugging
            decision_result: Optional[DecisionResponse] = None

            try:
                # Invoke the chain. The output is expected to be a DecisionResponse object.
                llm_raw_response = decision_chain.invoke(decision_prompt_input)

                decision_result = llm_raw_response # This should already be the parsed Pydantic model

                # --- 2d: Format Answer for HackRx ---
                answer_text = f"{decision_result.decision.capitalize()}: {decision_result.justification}"
                if decision_result.amount is not None:
                    answer_text += f" (Payout: ${decision_result.amount:,.2f})"
                
                answers.append(answer_text)
                print(f"Generated answer: '{answer_text[:80]}...'") # Log truncated answer

            except ValidationError as e:
                # Handle cases where the LLM's output doesn't match the DecisionResponse schema
                error_details = e.errors()
                print(f"API: Decision response validation error for question '{question}'. LLM Output: '{llm_raw_response if llm_raw_response else 'No raw output'}'. Errors: {error_details}")
                answers.append("Error: LLM output format mismatch for this question.")

            except Exception as e:
                # Catch other errors during the decision chain invocation or formatting
                print(f"API: Error processing decision for question '{question}'. LLM Output: '{llm_raw_response if llm_raw_response else 'No raw output'}'. Error: {e}")
                answers.append("Error: Failed to generate decision for this question.")

        except Exception as e:
            # Catch any unexpected errors during retrieval or overall processing for a question
            print(f"API: Unexpected error processing question '{question}': {e}")
            answers.append("Error: An unexpected error occurred for this question.")

    # --- Step 3: Log Total Time and Return Response ---
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n--- Finished processing {len(request_data.questions)} questions ---")
    print(f"Total execution time: {total_duration:.2f} seconds")
    
    return HackRxResponse(answers=answers)

# --- Root Endpoint (Optional, for health checks) ---
@app.get("/", summary="Health Check")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "API is running"}

# --- Initial Load Check ---
# This will trigger the LangchainManager initialization when the app starts.
# If the initialization fails due to missing keys, it will raise an error and the app won't start.
print("API starting up. Initializing Langchain components...")
# The LangchainManager's __init__ method is called automatically when the instance is created globally.
# If it fails during initialization, the RuntimeError exceptions will be caught by the runtime_error_handler
# or the process will exit if the error is not caught within __init__.
if not langchain_manager.vectorstore_loaded:
    print("Warning: Vector store not loaded at startup. It will be created upon the first request with a document URL.")

# --- To run this API locally for testing ---
# 1. Save this entire code block as `api.py`.
# 2. Install dependencies:
#    pip install fastapi uvicorn python-dotenv requests langchain-core langchain-community langchain-openai langchain-chroma "unstructured[pdf]" pdfminer.six python-docx pydantic
# 3. Create a .env file in the same directory with your API keys:
#    OPENAI_API_KEY=sk-...
#    HACKRX_API_KEY=your_secret_api_key_token
# 4. (Optional) Place a 'policy.pdf' in the same directory if you want to test initial setup without a URL.
# 5. Run the server from your terminal in the project's root directory:
#    uvicorn api:app --reload --host 0.0.0.0 --port 8000
# 6. Test using tools like Postman or curl, sending POST requests to http://localhost:8000/hackrx/run
#    Remember to include the 'Authorization: Bearer YOUR_HACKRX_API_KEY' header.