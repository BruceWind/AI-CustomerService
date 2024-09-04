from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import uvicorn
from document_processor import process_document
from chat_handler import handle_chat_message
from fastapi import Form

app = FastAPI()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None)
):
    try:
        result = await process_document(file, doc_type)
        return {"message": "Document processed successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
async def chat(message: str = Form(...)):
    return StreamingResponse(handle_chat_message(message), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
