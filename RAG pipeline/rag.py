# app/rag.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "vectorstore/faiss_index"

def ask_question(question: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    llm = Ollama(model="llama3")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant.
Answer ONLY using the context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    return {
        "answer": response,
        "sources": [d.metadata for d in docs]
    }
