import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(pdf_bytes):
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_bytes):
    text = ""
    with open("temp.docx", "wb") as f:
        f.write(docx_bytes)
    doc = docx.Document("temp.docx")
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text
