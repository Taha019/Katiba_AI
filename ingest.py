import os
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PDF_FOLDER = "./pdf_docs"
CHROMA_PATH = "./chroma_db"

def main():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created '{PDF_FOLDER}'. Place your PDF files there and run again.")
        return

    print(f"Scanning '{PDF_FOLDER}' for PDF files...")
    documents = []
    for pdf_path in sorted(Path(PDF_FOLDER).rglob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages):
            documents.append(Document(
                page_content=page.extract_text() or "",
                metadata={"source": str(pdf_path), "page": page_number}
            ))

    if not documents:
        print(f"No PDFs found in '{PDF_FOLDER}'. Exiting.")
        return

    print(f"Loaded {len(documents)} total pages across all PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(model="bge-m3")

    print("Generating embeddings and persistence layer in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    main()