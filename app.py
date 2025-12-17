import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp

# -----------------------------
# CONFIG
# -----------------------------
INDEX_PATH = "rag_index_neuro"
MODEL_PATH = "models/llama-2-7b-chat.gguf"  # ejemplo

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="RAG Neurofisiología", layout="centered")
st.title("📚 Asistente de estudio – Neurofisiología")

st.markdown("""
Sistema RAG para consulta académica basada exclusivamente en papers cargados.
""")

# -----------------------------
# CARGA EMBEDDINGS + FAISS
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# -----------------------------
# LLM LOCAL
# -----------------------------
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        temperature=0,
        verbose=False
    )

llm = load_llm()

# -----------------------------
# FUNCIONES
# -----------------------------
def recuperar_contexto(pregunta, k=4):
    docs = vectorstore.similarity_search(pregunta, k=k)
    return "\n\n".join([d.page_content for d in docs])

def responder(pregunta):
    contexto = recuperar_contexto(pregunta)

    prompt = f"""
Respondé SIEMPRE en español claro y académico.
Usá exclusivamente la información del contexto.
Si la respuesta no está en el contexto, decí:
"No se encuentra en los documentos analizados".

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""

    return llm.invoke(prompt)

# -----------------------------
# UI INTERACTIVA
# -----------------------------
pregunta = st.text_input("🧠 Escribí tu pregunta:")

if pregunta:
    with st.spinner("Analizando documentos..."):
        respuesta = responder(pregunta)

    st.subheader("📖 Respuesta")
    st.write(respuesta)
