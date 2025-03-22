import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set OpenRouter API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY. Please set it in the .env file.")

# Define request model
class ChatRequest(BaseModel):
    question: str

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(
    model="deepseek/deepseek-r1-zero:free",  # Free DeepSeek model on OpenRouter
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",  # Required for OpenRouter
    model_kwargs={  # Pass extra headers inside model_kwargs
        "extra_headers": {
            "HTTP-Referer": "https://yourwebsite.com",  # Change to your site URL
            "X-Title": "YourAppTitle"
        }
    }
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
    return {"message": "FastAPI is running! Use the /chat endpoint to interact."}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles chat requests."""
    try:
        response = chain.invoke({"question": request.question})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
