from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from parser import extract_text_from_pdf, extract_text_from_docx
from chunker import chunk_text
from retriever import build_faiss_index, get_similar_chunks
from answerer import answer_query
from utils import download_file_from_url
from typing import List
import json

app = FastAPI()

@app.post("/hackrx/run")
async def run_query(
    questions: str = Form(...),
    file: UploadFile = File(None),
    file_url: str = Form(None)
):
    try:
        questions = json.loads(questions)
        if not isinstance(questions, list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in questions field.")

    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either file or file_url must be provided.")

    # Read document
    if file:
        content = await file.read()
        extension = file.filename.split(".")[-1].lower()
    else:
        content, extension = download_file_from_url(file_url)

    if extension == "pdf":
        text = extract_text_from_pdf(content)
    elif extension == "docx":
        text = extract_text_from_docx(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)

    answers = []
    for question in questions:
        relevant_chunks = get_similar_chunks(chunks, index, embeddings, question)
        answer = answer_query(relevant_chunks, question)
        answers.append(answer)

    return JSONResponse(content={"answers": answers})
