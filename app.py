from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import uvicorn
from document_processor import process_document
from chat_handler import handle_chat_message
from fastapi import Form
import uuid

app = FastAPI()

rag_instances = {}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None)
):
    try:
        lib_id = str(uuid.uuid4())
        result = await process_document(file, doc_type)
        rag_instances[lib_id] = result['rag_engine']
        return {"message": "Document processed successfully", "lib_id": lib_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(message: str = Form(...), lib_id: str = Form(...)):
    if lib_id not in rag_instances:
        raise HTTPException(status_code=404, detail="lib id not found")
    return StreamingResponse(handle_chat_message(message, rag_instances[lib_id]), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
