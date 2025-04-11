import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY. Please set it in the .env file.")

# Define request model
class ChatRequest(BaseModel):
    question: str

# Initialize LangChain ChatGroq with LLaMA-4 Maverick
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct", 
    groq_api_key=groq_api_key
)

# Define prompt and processing pipeline
output_parser = StrOutputParser()
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "Question: {question}")
]) | llm | output_parser

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "FastAPI is running with LLaMA-4 Maverick on Groq! Use the /chat endpoint."}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles chat requests."""
    try:
        response = chain.invoke({"question": request.question})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
