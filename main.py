import os
import fitz
import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- BASIC SETUP ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- GLOBAL VARIABLES ---
# The RAG chain will be loaded on startup and stored here
qa_chain = None

# --- PDF PROCESSING LOGIC (from your reference) ---
def process_pdf_smartly(pdf_path: str) -> list[Document]:
    """
    Processes a PDF, extracting text and converting tables to Markdown format.
    This function is adapted from your provided reference code.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise FileNotFoundError(f"The PDF file was not found at {pdf_path}")

    doc = fitz.open(pdf_path)
    page_documents = []
    logger.info(f"Starting smart processing for {len(doc)} pages...")

    for page_num, page in enumerate(doc):
        # Extract tables and their locations first
        tables = page.find_tables()
        table_areas = [fitz.Rect(t.bbox) for t in tables] if tables else []
        page_content = ""

        # Process and format tables as Markdown
        for i, table in enumerate(tables or []):
            try:
                table_data = table.extract()
                if not table_data or not any(table_data): continue
                # Filter out None rows and cells
                valid_table_data = [row for row in table_data if row is not None and any(cell is not None for cell in row)]
                if not valid_table_data: continue

                header = " | ".join([str(h).replace('\n', ' ') for h in valid_table_data[0]])
                divider = " | ".join(["---"] * len(valid_table_data[0]))
                rows = "\n".join([" | ".join([str(cell).replace('\n', ' ') for cell in row]) for row in valid_table_data[1:]])
                markdown_table = f"\n\n--- TABLE START ---\n{header}\n{divider}\n{rows}\n--- TABLE END ---\n\n"
                page_content += markdown_table
            except Exception as e:
                logger.warning(f"Skipping a malformed table on page {page_num + 1}: {e}")

        # Process text blocks, avoiding those that are part of a table
        text_blocks = page.get_text("blocks", sort=True)
        for block in text_blocks:
            block_rect = fitz.Rect(block[:4])
            # Check if the block is inside any of the identified table areas
            is_in_table = any(block_rect.intersects(area) for area in table_areas)
            if not is_in_table:
                text = block[4]
                # Add a space for line breaks within a block, and a newline for the block itself
                cleaned_text = text.replace('\n', ' ').strip()
                if cleaned_text:
                    page_content += cleaned_text + "\n"
        
        if page_content.strip():
            page_documents.append(Document(
                page_content=page_content,
                metadata={"source": pdf_path, "page": page_num + 1}
            ))
            
    logger.info(f"Smart processing complete. Extracted content from {len(page_documents)} pages.")
    return page_documents

# --- FASTAPI APPLICATION SETUP ---
app = FastAPI(
    title="Professional Multilingual RAG API",
    description="An advanced RAG system using a MultiQueryRetriever and intelligent PDF processing.",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    """
    This function runs when the FastAPI application starts.
    It loads the PDF, creates the vector store, and initializes the RAG chain.
    """
    global qa_chain
    logger.info("Server startup: Initializing RAG chain...")

    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return

    try:
        # 1. Process PDF
        docs = process_pdf_smartly(pdf_path)

        # 2. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            logger.error("Document chunking resulted in 0 chunks. Cannot proceed.")
            return
        logger.info(f"Created {len(chunks)} document chunks.")

        # 3. Create multilingual embeddings
        logger.info("Loading multilingual embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # 4. Create FAISS Vector Store
        logger.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        logger.info("FAISS vector store created successfully.")

        # 5. Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key, temperature=0)

        # 6. Set up the MultiQueryRetriever
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        multiquery_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        logger.info("MultiQueryRetriever initialized.")
        
        # 7. Define the Prompt Template
        reasoning_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert Q&A system for the Bengali study guide 'Aparichita'.
            Answer the user's question in Bengali based ONLY on the provided context.
            Your main goal is to be precise. If the question asks for a name, give only the name. If it asks for a number, give only the number.

            - To answer, first check the regular text.
            - If the answer isn't there, look for it in the multiple-choice questions (MCQs) or tables.
            - If you find the answer, state it concisely. Do not add extra words.

            Context:
            {context}

            Question:
            {question}

            Concise Answer:
            """
        )

        # 8. Create the final RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=multiquery_retriever,
            chain_type_kwargs={"prompt": reasoning_prompt},
            return_source_documents=True # Optionally return source documents
        )
        logger.info("✅ RAG chain is fully initialized and ready to accept queries.")

    except Exception as e:
        logger.exception(f"An error occurred during RAG chain initialization: {e}")

# --- API ENDPOINTS ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_page: int | None = None

@app.get("/", summary="Root Endpoint")
def read_root():
    return {"message": "Welcome to the Professional RAG API. Navigate to /docs for more info."}

@app.post("/ask", response_model=QueryResponse, summary="Ask a question")
async def ask_question(request: QueryRequest):
    """
    Accepts a user query in English or Bengali and returns a precise answer
    grounded in the provided PDF document.
    """
    if qa_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system is not initialized. Please check server logs for errors."
        )
    
    logger.info(f"Received query: {request.question}")
    
    try:
        response = qa_chain.invoke({"query": request.question})
        answer = response.get('result', "Could not determine an answer.").strip()
        
        # Extract source page if available
        source_page = None
        if response.get("source_documents"):
            source_page = response["source_documents"][0].metadata.get("page")

        logger.info(f"Generated Answer: '{answer}' from page {source_page}")
        return {"answer": answer, "source_page": source_page}

    except Exception as e:
        logger.exception(f"An error occurred while processing the query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )