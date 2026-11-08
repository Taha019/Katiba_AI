# Katiba AI

A local retrieval-augmented question-answering tool for the Constitution of Kenya. It indexes PDF documents, retrieves relevant passages, and uses a local Ollama model to answer questions with source-page references.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The `bge-m3` embedding model and `qwen3:8b` chat model available in Ollama

## Setup

Run these commands from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen3:8b
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Put the Constitution PDF or other source PDFs in `pdf_docs/`. The repository includes `The_Constitution_of_Kenya_2010.pdf` as an example source.

## Use

Build the local vector database from the PDFs:

```bash
python ingest.py
```

Then start the question-answering prompt:

```bash
python query.py
```

Enter a question at the prompt. Type `exit` to quit. Answers are generated from retrieved PDF passages, and the matching source files and page numbers are printed below each answer.

## Project files

- `ingest.py` extracts PDF text, splits it into overlapping chunks, creates embeddings, and stores them in `chroma_db/`.
- `query.py` retrieves the four most relevant chunks and asks `qwen3:8b` to answer using that context.
- `pdf_docs/` contains source PDFs.
- `chroma_db/` is the generated local vector database.

Run ingestion after adding source PDFs. The current ingestion script appends records to the existing Chroma database, so avoid rerunning it on the same files unless you intend to rebuild or update the index.