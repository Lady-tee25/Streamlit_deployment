import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")


# ============================================================
# CONFIGURATION
# ============================================================

# -------------------------
# Groq / Chat model
# -------------------------

CHAT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-20b"


# -------------------------
# Embedding model
# -------------------------

EMBEDDING_BASE_URL = "https://qwen-embed.publicaai.com/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


# -------------------------
# Existing vector store
# -------------------------
#
# app.py is assumed to be inside:
#
# Slac/
#   app.py
#   msme_store/
#       msme_index.json
#
# Therefore, we build the path relative to app.py.
#

STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "msme_store",
    "msme_index.json"
)


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Nigeria RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #0F766E;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748B;
        margin-bottom: 25px;
    }

    /* Source box */
    .source-box {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #0F766E;
        margin-top: 10px;
    }

    /* Chat input */
    .stChatInput {
        padding-bottom: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏥 Health Nigeria RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about health policies, healthcare, '
    'regulations, industry information, and related topics in Nigeria.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK API KEYS
# ============================================================

if not OPENAI_API_KEY:

    st.error(
        "OPENAI_API_KEY was not found.\n\n"
        "Please add your OpenAI API key to the .env file."
    )

    st.stop()


if not EMBEDDING_API_KEY:

    st.error(
        "EMBEDDING_API_KEY was not found.\n\n"
        "Please add your embedding API key to the .env file."
    )

    st.stop()


# ============================================================
# CHECK VECTOR STORE
# ============================================================

if not os.path.exists(STORE_PATH):

    st.error(
        f"""
Vector store not found.

Expected location:

{STORE_PATH}

Please make sure that:

1. The `msme_store` folder exists.
2. `msme_index.json` exists inside it.
3. Your indexing script has already created the vector store.
"""
    )

    st.stop()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )

    return embeddings


# ============================================================
# LOAD SKLEARN VECTOR STORE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = SKLearnVectorStore(
        embedding=embeddings,
        persist_path=STORE_PATH,
        serializer="json"
    )

    return vectorstore


# ============================================================
# LOAD RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )

    return retriever


# ============================================================
# LOAD CHAT MODEL
# ============================================================

@st.cache_resource
def load_chat_model():

    chat_model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=CHAT_BASE_URL,
        model=CHAT_MODEL,
        temperature=0
    )

    return chat_model


# ============================================================
# RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a knowledgeable data analyst providing insights about
Health in Nigeria.

Your task is to answer the user's question using ONLY the
information contained in the provided context.

The context may contain information about:

- Health in Nigeria
- Healthcare
- Health policies
- Healthcare regulations
- Government programs
- Healthcare organizations
- Health industry information
- Starting a health-related business
- Growing a health-related business
- Sustaining a health-related business
- Industry-specific information
- Relevant documents and sources

IMPORTANT RULES:

1. Use the provided context as the primary source of truth.

2. Do not invent facts.

3. Do not invent statistics.

4. Do not invent policies or regulations.

5. Do not invent URLs.

6. Do not claim something is in the documents if it is not.

7. If the answer cannot be found in the context, say:

   "I could not find this information in the available documents."

8. Answer the exact question asked by the user.

9. Be clear, professional, and concise while providing
   sufficient explanation.

10. Use Markdown formatting.

SOURCE RULES:

If the retrieved documents contain source information, provide
the relevant sources at the end of the answer.

If a URL is available, include:

**To read more:** [URL]

Do not create a URL if one is not provided in the context.

------------------------------------------------------------
CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""
)


# ============================================================
# CREATE RAG CHAIN
# ============================================================

@st.cache_resource
def load_chain():

    chat_model = load_chat_model()

    chain = (
        prompt
        | chat_model
        | StrOutputParser()
    )

    return chain


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

try:

    retriever = load_retriever()
    chain = load_chain()

except Exception as e:

    st.error(
        "There was a problem loading the vector store or "
        "embedding model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )

        # -----------------------------------------
        # Display sources
        # -----------------------------------------

        if message.get("sources"):

            with st.expander("📚 Retrieved Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"- `{source}`"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Health in Nigeria..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(question)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message("assistant"):

        try:

            # ----------------------------------------------
            # Retrieve documents
            # ----------------------------------------------

            with st.spinner(
                "🔎 Searching the knowledge base..."
            ):

                retrieved_docs = retriever.invoke(
                    question
                )


            # ----------------------------------------------
            # Check retrieval
            # ----------------------------------------------

            if not retrieved_docs:

                answer = (
                    "I could not find relevant information "
                    "in the available documents."
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": []
                    }
                )

            else:

                # ------------------------------------------
                # Build context
                # ------------------------------------------

                context_parts = []

                for doc in retrieved_docs:

                    if doc.page_content:

                        context_parts.append(
                            doc.page_content
                        )


                context = "\n\n---\n\n".join(
                    context_parts
                )


                # ------------------------------------------
                # Generate answer
                # ------------------------------------------

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = chain.invoke(
                        {
                            "context": context,
                            "question": question
                        }
                    )


                # ------------------------------------------
                # Display answer
                # ------------------------------------------

                st.markdown(answer)


                # ------------------------------------------
                # Extract sources
                # ------------------------------------------

                sources = []

                for doc in retrieved_docs:

                    metadata = doc.metadata or {}

                    # Try common metadata names
                    source = (
                        metadata.get("source")
                        or metadata.get("url")
                        or metadata.get("link")
                        or metadata.get("file_path")
                        or metadata.get("filepath")
                    )


                    if source:

                        if source not in sources:

                            sources.append(source)


                # ------------------------------------------
                # Display retrieved documents
                # ------------------------------------------

                with st.expander(
                    f"📖 Retrieved Documents ({len(retrieved_docs)})"
                ):

                    for i, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        st.markdown(
                            f"### Document {i}"
                        )

                        metadata = doc.metadata or {}

                        if metadata:

                            st.caption(
                                f"Metadata: {metadata}"
                            )

                        st.write(
                            doc.page_content[:2000]
                        )

                        if i < len(retrieved_docs):

                            st.divider()


                # ------------------------------------------
                # Display sources
                # ------------------------------------------

                if sources:

                    with st.expander(
                        "🔗 Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"- {source}"
                            )


                # ------------------------------------------
                # Save assistant response
                # ------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


        except Exception as e:

            error_message = (
                "Sorry, something went wrong while "
                "processing your question."
            )

            st.error(error_message)

            st.exception(e)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")


    # --------------------------------------------------------
    # Vector store
    # --------------------------------------------------------

    st.markdown("### 📦 Vector Store")

    st.write(
        "SKLearnVectorStore"
    )

    st.caption(
        "No Chroma database is used."
    )


    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    st.markdown("### 📁 Index")

    st.code(
        STORE_PATH,
        language="text"
    )


    # --------------------------------------------------------
    # Chat model
    # --------------------------------------------------------

    st.markdown("### 🤖 Chat Model")

    st.code(
        CHAT_MODEL,
        language="text"
    )


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    st.markdown("### 🧠 Embeddings")

    st.code(
        EMBEDDING_MODEL,
        language="text"
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    st.markdown("### 🔎 Retrieval")

    st.write(
        "Method: MMR"
    )

    st.write(
        "Documents: 4"
    )

    st.write(
        "Fetch candidates: 10"
    )


    st.divider()


    # --------------------------------------------------------
    # Clear conversation
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    # --------------------------------------------------------
    # About
    # --------------------------------------------------------

    st.divider()

    st.markdown("### About")

    st.caption(
        "This chatbot uses retrieval-augmented generation "
        "(RAG) to answer questions using the documents "
        "stored in the local SKLearnVectorStore index."
    )
