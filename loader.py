import pandas as pd
from llama_index.core import Document
import os

def load_document(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file)
        text_data = "\n".join(df.astype(str).apply(lambda row: " ".join(row), axis=1))
    elif ext in [".txt", ".md"]:
        text_data = file.read().decode("utf-8")
    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text_data = "\n".join(page.extract_text() for page in reader.pages)
    else:
        raise ValueError("Unsupported file type.")
    return [Document(text=text_data)]
