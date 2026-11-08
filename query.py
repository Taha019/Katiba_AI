import sys
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_PATH = "./chroma_db"

def build_rag_components():
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)
    return retriever, llm

def main():
    try:
        retriever, llm = build_rag_components()
    except Exception as e:
        print(f"Error loading vector DB or LLM: {e}")
        sys.exit(1)

    print("\n--- PDF RAG System Ready (Qwen3:8b) ---")
    while True:
        query = input("\nAsk a question about your PDFs (or 'exit' to quit): ")
        if query.strip().lower() == 'exit':
            break
        if not query.strip():
            continue

        source_documents = retriever.invoke(query)
        context = "\n\n".join(document.page_content for document in source_documents)
        response = llm.invoke([
            SystemMessage(content=(
                "Answer using only the supplied context. If the context does not "
                "contain the answer, say you don't know."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ])

        print("\n=== Response ===")
        print(response.content)

        print("\n=== Sources ===")
        for doc in source_documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            print(f"- {source} (Page {page})")

if __name__ == "__main__":
    main()