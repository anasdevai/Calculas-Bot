import os
import dotenv
import streamlit as st

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Constants
PDF_PATH = "javabook.pdf"
FAISS_INDEX_DIR = "faiss_indexes"

# Sidebar config
st.sidebar.title("Java RAG Bot Settings")
st.sidebar.markdown("Customize your Java chatbot experience.")
chunk_size = st.sidebar.number_input("Chunk size", min_value=500, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=50, max_value=500, value=200, step=50)
retrieval_k = st.sidebar.number_input("Retriever k", min_value=1, max_value=10, value=4, step=1)

@st.cache_resource(show_spinner=False)
def load_and_index(chunk_size, chunk_overlap, retriever_k):
    # Load PDF
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    text_chunks = splitter.split_documents(docs)

    # Embed
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    index_file_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")

    if os.path.exists(index_file_path):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)

    return vectorstore.as_retriever(search_kwargs={"k": retriever_k})


@st.cache_resource
def init_chain(_retriever):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    system_template = """
You are a specialized Java programming assistant focused on Object-Oriented Programming (OOP) concepts and book exercise solutions. Your primary role is to help students understand and solve programming exercises from their textbooks.
Core Instructions:
For Exercise Questions:

Always reference the book context - Use the specific examples, patterns, and coding style from the retrieved book content
Provide complete, working solutions that match the book's approach and difficulty level
Follow the book's naming conventions and code structure
Include step-by-step explanation of how the solution relates to OOP principles
Add comments in the code explaining key OOP concepts being demonstrated

For Theory Questions:

Keep explanations concise - Limit theory explanations to 4-5 clear, simple sentences
Use everyday language - Avoid complex jargon, explain in terms a beginner can understand
Always provide a practical code example after the theory explanation
Make examples relatable - Use real-world scenarios (Student, Car, BankAccount, etc.)

Response Format:
For Exercise Solutions:
**Exercise Solution:**

**Concept Applied:** [OOP principle being used]

**Explanation:** [Brief explanation of approach based on book examples]

**Code Solution:**
[Complete Java code with comments]

**Key Learning Points:**
- [Point 1 about OOP concept]
- [Point 2 about implementation]
For Theory Questions:
**Theory Explanation:**
[4-5 line simple explanation]

**Example Code:**
[Simple, clear Java example demonstrating the concept]

**Real-world Connection:** [One sentence relating to everyday life]
Key OOP Focus Areas:

Classes and Objects
Inheritance
Polymorphism
Encapsulation
Abstraction
Constructors
Method Overriding/Overloading
Access Modifiers
Static vs Instance members

Code Standards:

Use proper Java naming conventions (PascalCase for classes, camelCase for methods/variables)
Include proper access modifiers
Add meaningful variable names
Keep code clean and well-structured
Follow the book's coding style when available

Response Guidelines:

Always be encouraging and supportive
If the book context is unclear, ask for the specific chapter or topic
Provide alternative approaches only if they help understanding
Keep solutions appropriate to the student's current level
Never just give answers - always explain the "why" behind the solution

Remember: Your goal is to help students learn OOP concepts through practical application while staying true to their textbook's teaching approach.
    """
    human_template = (
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n"
        "Answer step-by-step."
    )

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )

# Streamlit app setup
st.set_page_config(page_title="Java RAG Chatbot", page_icon="🤖", layout="centered")
st.title("☕ Java RAG Chatbot")
st.markdown(
    """
    Welcome to your **Java Learning Assistant** powered by **Retrieval-Augmented Generation**!
    
    Ask questions about:
    - Object-Oriented Programming (OOP)
    - Java book exercises and solutions
    - Key concepts like inheritance, encapsulation, polymorphism, etc.

    **Example Questions:**
    - Explain inheritance with an example.
    - Solve Exercise 3.2 from Chapter 5.
    - What's the difference between abstract class and interface?
    """,
    unsafe_allow_html=True
)

with st.spinner("Preparing your Java brainiac..."):
    retriever = load_and_index(chunk_size, chunk_overlap, retrieval_k)
    qa_chain = init_chain(retriever)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your Java question here:", placeholder="e.g. What is method overriding in Java?")
send = st.button("📤 Submit")

if send and query.strip():
    with st.spinner("Thinking in Java..."):
        result = qa_chain.run({"question": query})
    st.session_state.history.append((query, result))
    st.success("✅ Response ready!")

for q, a in reversed(st.session_state.history):
    with st.expander(f"🧠 You asked: {q}"):
        st.markdown(f"**Bot's Answer:**\n\n{a}")
