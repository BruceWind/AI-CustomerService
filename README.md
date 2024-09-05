# RAG-Powered Chat Application

This project implements a Retrieval-Augmented Generation (RAG) system with OpenAI integration, featuring document upload capabilities and a streaming chat API.

## Features

- Document upload and processing for RAG
- Streaming chat API with OpenAI integration
- FastAPI-based web server

## Installation

1.  Install **tesseract-ocr** and Clone the repository.

- For macOS, use Homebrew: `brew install tesseract`
- For ubuntu: `sudo apt-get update && sudo apt-get install tesseract-ocr`


2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation faiss-cpu==1.7.2
```

<details>

<summary>
 Click to expend: If you suffer an error: `Failed to build faiss-cpu` from this step:
</summary>
   The error is related to building the faiss-cpu package, which requires SWIG. Here's a quick guide to resolve this issue:
1. Install SWIG:

   - For macOS, use Homebrew: `brew install swig`
   - For ubuntu: `sudo apt-get update && sudo apt-get install build-essential swig libopenblas-dev`

 2. or, SWIG does not work
   - you can try using Anaconda/Miniconda. However, I dont want to put much tutorial here.
   


2. After installing SWIG, try installing the requirements again:
   ```
   pip install -r requirements.txt
   ```

3. If issues persist with faiss-cpu, try using a pre-built wheel:
   - In requirements.txt, replace `faiss-cpu==1.7.2` with `faiss-cpu==1.7.2 --only-binary :all:`

4. Alternatively, use faiss-cpu from conda:
   - Install Miniconda or Anaconda
   - Create a new conda environment: `conda create -n your_env_name python=3.8`
   - Install faiss-cpu: `conda install -c conda-forge faiss-cpu`
   - Install other requirements: `pip install -r requirements.txt`
</details>


3. Start the server:

    > Create a .env file:
   Create a file named `.env` in the root directory of the project and add the following content:
   ```
   OPENAI_API_BASE=https://XXXXX/v1
   OPENAI_API_KEY=your_api_key_here
   ```

Following that, run this command:

```
python -m venv venv
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server will run on `http://localhost:8000`

## API Endpoints:
   - POST `/upload`: Upload a document for processing
     - Parameters:
       - `file`: The document file to upload
       - `doc_type`: (Optional) The type of the document
   - POST `/chat`: Send a chat message and receive a streaming response
     - Parameters:
       - `message`: The chat message to process


## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://openai.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [RAG (Retrieval-Augmented Generation)](https://arxiv.org/abs/2005.11401)
