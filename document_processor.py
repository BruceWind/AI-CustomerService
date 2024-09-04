from fastapi import UploadFile
import os
from typing import Optional
import PyPDF2
from docx import Document
from rag_engine import rag_engine
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io

async def process_document(file: UploadFile, doc_type: Optional[str]):
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Save the uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Extract text based on file type
    text = extract_text(file_path, doc_type)
    
    # Add the extracted text to the RAG engine
    await rag_engine.add_documents([text])
    
    return {
        "file_path": file_path,
        "doc_type": doc_type,
        "file_size": len(content),
        "file_name": file.filename,
        "text_length": len(text)
    }

def extract_text(file_path: str, doc_type: Optional[str]) -> str:
    if doc_type == "pdf" or file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif doc_type == "docx" or file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        # Assume it's a plain text file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    # First, try to extract text directly from the PDF
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() for page in reader.pages)
    
    # If no text was extracted, assume the PDF contains images
    if not text.strip():
        # Convert PDF to images
        images = convert_from_path(file_path)
        # Perform OCR on each image
        for image in images:
            text += pytesseract.image_to_string(image) + " "
    
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return " ".join(paragraph.text for paragraph in doc.paragraphs)