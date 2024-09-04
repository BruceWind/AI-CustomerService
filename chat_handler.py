import asyncio
from openai import AsyncOpenAI
import os
import nltk
from nltk.corpus import wordnet
from rag_engine import rag_engine
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Set up your OpenAI API configuration
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

def expand_query(query: str) -> str:
    expanded_terms = []
    for word in query.split():
        synonyms = wordnet.synsets(word)
        if synonyms:
            expanded_terms.append(synonyms[0].lemmas()[0].name())
        else:
            expanded_terms.append(word)
    return " ".join(expanded_terms)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

async def handle_chat_message(message: str):
    expanded_message = expand_query(message)
    relevant_docs = await rag_engine.query(expanded_message)
    
    context = " ".join(doc['content'] for doc in relevant_docs)
    
    # Limit context to approximately 6000 tokens (leaving room for the system message and user question)
    while num_tokens_from_string(context, "cl100k_base") > 6000:
        context = context[:int(len(context)*0.9)]  # Reduce context by 10% each iteration
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {message}"}
    ]
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=1000  # Limit the response length
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield f"data: {chunk.choices[0].delta.content}\n\n"
        await asyncio.sleep(0.1)