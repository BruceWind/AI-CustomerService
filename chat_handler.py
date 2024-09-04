import asyncio
from openai import AsyncOpenAI
import os
from rag_engine import rag_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up your OpenAI API configuration
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

async def handle_chat_message(message: str):
    # Retrieve relevant documents using RAG
    relevant_docs = rag_engine.query(message)
    
    # Prepare the messages for the chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context: {' '.join(doc['content'] for doc in relevant_docs)}\n\nQuestion: {message}"}
    ]
    
    # Create a streaming chat completion
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",  # Changed model to gpt4o-mini
        messages=messages,
        stream=True
    )
    
    # Stream the response
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
        await asyncio.sleep(0.1)  # Small delay to control streaming rate